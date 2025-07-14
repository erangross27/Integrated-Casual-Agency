# ICA Framework Continuous Learning Service (PowerShell)
# Keeps learning running with better error handling and logging

param(
    [int]$MaxRestarts = 100,
    [int]$RestartDelay = 5
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ICA Framework - Continuous Learning Service" -ForegroundColor Cyan  
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Max restarts: $MaxRestarts" -ForegroundColor Yellow
Write-Host "Restart delay: $RestartDelay seconds" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

$restartCount = 0

try {
    while ($restartCount -lt $MaxRestarts) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "[$timestamp] Starting learning session... (Restart #$restartCount)" -ForegroundColor Green
        
        # Run the continuous learning
        $process = Start-Process -FilePath "python" -ArgumentList "run_continuous.py" -Wait -PassThru -NoNewWindow
        
        $exitCode = $process.ExitCode
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        
        if ($exitCode -eq 0) {
            Write-Host "[$timestamp] Session completed normally" -ForegroundColor Blue
        } else {
            Write-Host "[$timestamp] Session ended with exit code: $exitCode" -ForegroundColor Red
        }
        
        $restartCount++
        
        if ($restartCount -lt $MaxRestarts) {
            Write-Host "[$timestamp] Restarting in $RestartDelay seconds..." -ForegroundColor Yellow
            Start-Sleep -Seconds $RestartDelay
        }
    }
    
    Write-Host "Maximum restart limit reached ($MaxRestarts)" -ForegroundColor Red
}
catch {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] Service interrupted: $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Write-Host "Continuous learning service stopped" -ForegroundColor Cyan
}
