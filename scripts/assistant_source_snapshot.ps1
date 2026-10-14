param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($ProjectRoot)
$baselineRoot = Join-Path $root '.assistant-baselines'
$excludedDirectoryNames = @('__pycache__', 'logs', 'models', '.cache', 'cache', 'caches', 'recordings', 'token_tracker', '.venv', 'venv', 'venvs')
$maximumFiles = 5000
$maximumFileBytes = 5MB
$maximumTotalBytes = 100MB
$scriptExtensions = @('.ps1', '.py', '.bat', '.cmd')
$fixedFiles = @(
    'assistant/config.toml',
    'assistant/requirements.txt',
    'pytest.ini',
    'README.md',
    '.gitignore',
    'AGENTS.md',
    'ASSISTANT_ARCHITECTURE.md',
    'ASSISTANT_PLAN.md'
)

function Get-ContainedRelativePath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $prefix = $root.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes project root: $Path"
    }
    return $full.Substring($prefix.Length).Replace('\', '/')
}

function Test-ReparsePoint([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    return (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Test-ExcludedDirectory([string]$Name) {
    return (($excludedDirectoryNames -contains $Name) -or ($Name -like '.venv-*'))
}

function Get-SourceFiles([string]$RelativeDirectory, [string[]]$Extensions) {
    $start = Join-Path $root $RelativeDirectory
    if (-not (Test-Path -LiteralPath $start -PathType Container)) { return @() }
    if (Test-ReparsePoint $start) { throw "Reparse point encountered at selected source root: $start" }
    $pending = [System.Collections.Generic.Stack[string]]::new()
    $pending.Push($start)
    $found = [System.Collections.Generic.List[string]]::new()
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in Get-ChildItem -LiteralPath $directory -Force) {
            if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse point encountered in selected source tree: $($item.FullName)"
            }
            if ($item.PSIsContainer) {
                if (-not (Test-ExcludedDirectory $item.Name)) { $pending.Push($item.FullName) }
            } elseif ($Extensions -contains $item.Extension.ToLowerInvariant()) {
                $found.Add($item.FullName)
            }
        }
    }
    return $found.ToArray()
}

# Check the project path as it exists now, including its local ancestors. This is
# a bounded check and does not prevent a path from changing between checks.
$cursor = Get-Item -LiteralPath $root -Force
while ($null -ne $cursor) {
    if (($cursor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Reparse point encountered in project path: $($cursor.FullName)"
    }
    $cursor = $cursor.Parent
}

$files = [System.Collections.Generic.List[string]]::new()
$files.AddRange([string[]](Get-SourceFiles 'assistant' @('.py')))
$files.AddRange([string[]](Get-SourceFiles 'tests' @('.py')))
$files.AddRange([string[]](Get-SourceFiles 'scripts' $scriptExtensions))
foreach ($relative in $fixedFiles) {
    $path = Join-Path $root $relative
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse point encountered for selected file: $path"
        }
        $files.Add($path)
    }
}

$uniqueFiles = @($files | Sort-Object -Unique)
if ($uniqueFiles.Count -gt $maximumFiles) {
    throw "Selected source file count $($uniqueFiles.Count) exceeds limit $maximumFiles; no archive was created."
}
$totalBytes = 0L
foreach ($file in $uniqueFiles) {
    $size = (Get-Item -LiteralPath $file -Force).Length
    if ($size -gt $maximumFileBytes) {
        throw "Selected file '$file' is $size bytes, exceeding the $maximumFileBytes-byte limit; no archive was created."
    }
    $totalBytes += [long]$size
    if ($totalBytes -gt $maximumTotalBytes) {
        throw "Selected source total exceeds $maximumTotalBytes bytes; no archive was created."
    }
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssffffZ')
if (Test-Path -LiteralPath $baselineRoot) {
    if (Test-ReparsePoint $baselineRoot) { throw "Baseline output directory is a reparse point: $baselineRoot" }
} else {
    New-Item -ItemType Directory -Path $baselineRoot -ErrorAction Stop | Out-Null
    if (Test-ReparsePoint $baselineRoot) { throw "Baseline output directory became a reparse point: $baselineRoot" }
}
$destination = Join-Path $baselineRoot $stamp
New-Item -ItemType Directory -Path $destination -ErrorAction Stop | Out-Null
$archivePath = Join-Path $destination 'assistant-source.zip'
$manifestPath = Join-Path $destination 'manifest.json'

try {
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $stream = [System.IO.File]::Open($archivePath, [System.IO.FileMode]::CreateNew)
    try {
        $archive = [System.IO.Compression.ZipArchive]::new($stream, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            $manifestFiles = [System.Collections.Generic.List[object]]::new()
            foreach ($file in $uniqueFiles) {
                $relative = Get-ContainedRelativePath $file
                $bytes = [System.IO.File]::ReadAllBytes($file)
                $hasher = [System.Security.Cryptography.SHA256]::Create()
                try { $sha = [BitConverter]::ToString($hasher.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant() } finally { $hasher.Dispose() }
                $entry = $archive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
                $entryStream = $entry.Open()
                try { $entryStream.Write($bytes, 0, $bytes.Length) } finally { $entryStream.Dispose() }
                $manifestFiles.Add([ordered]@{ path = $relative; size_bytes = $bytes.Length; sha256 = $sha })
            }
        } finally { $archive.Dispose() }
    } finally { $stream.Dispose() }
    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $manifest = [ordered]@{
        status = 'complete'
        created_utc = [DateTime]::UtcNow.ToString('o')
        project_root = $root
        archive = [ordered]@{ file = 'assistant-source.zip'; sha256 = $archiveHash }
        file_count = $manifestFiles.Count
        total_source_bytes = $totalBytes
        exclusions = @('.env and credentials', '.cache/cache/caches directories', 'assistant/logs', 'assistant/models', '__pycache__', 'recordings directories', 'token_tracker directories', '.venv and .venv-* directories', 'venv and venvs directories', 'personal documents', 'non-source files under scripts')
        files = $manifestFiles
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    [pscustomobject]@{ Snapshot = $destination; Files = $manifestFiles.Count; Bytes = $totalBytes; ArchiveSha256 = $archiveHash }
} catch {
    throw "Snapshot failed in '$destination': $($_.Exception.Message)"
}
