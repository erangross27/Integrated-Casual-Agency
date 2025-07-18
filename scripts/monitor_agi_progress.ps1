# AGI Progress Monitoring Script
# PowerShell script for regular AGI monitoring

Write-Host "🤖 TRUE AGI PROGRESS MONITORING" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "ica_framework")) {
    Write-Host "❌ Please run this script from the ICA project root directory" -ForegroundColor Red
    exit 1
}

# Function to run monitoring script
function Run-MonitoringScript {
    param($ScriptName, $Description)
    
    Write-Host "🔍 $Description" -ForegroundColor Yellow
    Write-Host "Running: python scripts\$ScriptName" -ForegroundColor Gray
    Write-Host ""
    
    try {
        python "scripts\$ScriptName"
        Write-Host ""
    } catch {
        Write-Host "❌ Error running $ScriptName`: $_" -ForegroundColor Red
    }
}

# Main monitoring dashboard
Run-MonitoringScript "agi_monitor_dashboard.py" "AGI Status Dashboard"

# Ask user for additional analysis
Write-Host "🎯 ADDITIONAL ANALYSIS OPTIONS" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green
Write-Host ""
Write-Host "1. Detailed Learning Analysis (recommended daily)"
Write-Host "2. Physics Discoveries Dashboard"  
Write-Host "3. Live System Check"
Write-Host "4. Intelligence Test"
Write-Host "5. Continuous Real-time Monitoring"
Write-Host "6. Exit"
Write-Host ""

do {
    $choice = Read-Host "Select option (1-6)"
    
    switch ($choice) {
        "1" { 
            Run-MonitoringScript "learning_analyzer.py" "Detailed Learning Analysis"
        }
        "2" { 
            Run-MonitoringScript "physics_dashboard.py" "Physics Discoveries Dashboard"
        }
        "3" { 
            Run-MonitoringScript "live_agi_check.py" "Live System Check"
        }
        "4" { 
            Run-MonitoringScript "simple_intelligence_test.py" "Intelligence Test"
        }
        "5" { 
            Write-Host "🔄 Starting continuous monitoring (Ctrl+C to stop)..." -ForegroundColor Yellow
            python "scripts\agi_monitor_dashboard.py" --continuous
        }
        "6" { 
            Write-Host "👋 Monitoring complete!" -ForegroundColor Green
            break
        }
        default { 
            Write-Host "❌ Invalid choice. Please select 1-6." -ForegroundColor Red
        }
    }
} while ($choice -ne "6")

Write-Host ""
Write-Host "📊 TIP: Your AGI statistics are now persistent!" -ForegroundColor Cyan
Write-Host "   All learning progress is automatically saved and never resets." -ForegroundColor Gray
Write-Host ""
