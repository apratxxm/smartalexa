# Register Memoir as a Scheduled Task that starts at logon.
#
# A scheduled task rather than a Startup-folder shortcut, because it can restart
# itself on failure -- which matters for a process you are supposed to forget
# about for months at a time.
#
# Run from an elevated PowerShell:  .\scripts\install_task.ps1

$ErrorActionPreference = "Stop"

$root      = Split-Path -Parent $PSScriptRoot
$pythonw   = Join-Path $root ".venv\Scripts\pythonw.exe"   # no console window
$taskName  = "Memoir Assistant"

if (-not (Test-Path $pythonw)) {
    throw "Not found: $pythonw. This script only registers the scheduled task. From the project root run: python -m venv .venv; .venv\Scripts\python.exe -m pip install -r assistant\requirements.txt"
}

$action = New-ScheduledTaskAction -Execute $pythonw `
    -Argument "-m assistant.supervise" -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Restart on failure, and never stop it for running too long or on battery --
# an always-on room assistant that quietly dies on battery is useless.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Always-on local voice control for Spotify." `
    -Force | Out-Null

Write-Host "Registered '$taskName'." -ForegroundColor Green
Write-Host "  Start now:  Start-ScheduledTask -TaskName '$taskName'"
Write-Host "  Stop:       Stop-ScheduledTask  -TaskName '$taskName'"
Write-Host "  Remove:     Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
Write-Host "  Logs:       $root\assistant\logs\memoir.log"
