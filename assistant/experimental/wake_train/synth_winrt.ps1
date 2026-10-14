# Local, offline TTS via the Windows OneCore voices (WinRT SpeechSynthesizer).
# Generates short wavs of a phrase list for every installed voice x prosody variant.
# Usage: powershell -File synth_winrt.ps1 -OutDir <dir> -Phrases "Alexa|Alexa." -Tag pos
param([string]$OutDir, [string]$Phrases, [string]$Tag, [int]$MaxVariants = 6)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, $type) {
    $t = $asTaskGeneric.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voices = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices
$rates = @('x-slow', 'slow', 'medium', 'fast', 'x-fast', 'medium')
$pitches = @('-15%', '+0%', '+10%', '-5%', '+15%', '-25%')
$pl = $Phrases -split '\|'
$n = 0
foreach ($v in $voices) {
    $synth.Voice = $v
    $vi = 0
    foreach ($ph in $pl) {
        for ($k = 0; $k -lt $MaxVariants; $k++) {
            $r = $rates[$k % $rates.Count]; $p = $pitches[($k + $vi) % $pitches.Count]
            $lang = $v.Language
            $esc = [System.Security.SecurityElement]::Escape($ph)
            $ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='$lang'><prosody rate='$r' pitch='$p'>$esc</prosody></speak>"
            try {
                $stream = Await ($synth.SynthesizeSsmlToStreamAsync($ssml)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
                $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
                $null = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
                $bytes = New-Object byte[] $stream.Size
                $reader.ReadBytes($bytes)
                $safe = ($v.DisplayName -replace '[^A-Za-z0-9]', '')
                [System.IO.File]::WriteAllBytes((Join-Path $OutDir ("{0}_{1}_{2:000}_{3}.wav" -f $Tag, $safe, $vi, $k)), $bytes)
                $n++
            } catch { Write-Output ("fail {0} {1}: {2}" -f $v.DisplayName, $ph, $_.Exception.Message) }
        }
        $vi++
    }
}
Write-Output "wrote $n files; voices: $(($voices | ForEach-Object { $_.DisplayName + '/' + $_.Language }) -join ', ')"
