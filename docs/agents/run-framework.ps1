<#
.SYNOPSIS
    Claude Code Agents Framework Runner (Windows PowerShell)

.DESCRIPTION
    Runs the Claude Code Agents Framework prompts in sequence.
    Claude CLI runs from the project root where CLAUDE.md and .claude/ folder exist.

.PARAMETER Step
    Run a specific step by number (e.g., 1, 4.5, 4.6)

.PARAMETER From
    Start step number for range execution

.PARAMETER To
    End step number for range execution

.PARAMETER All
    Run all steps without pausing between them

.PARAMETER List
    List all available steps

.EXAMPLE
    .\run-framework.ps1
    Run all steps interactively (pause between steps)

.EXAMPLE
    .\run-framework.ps1 -All
    Run all steps without pausing

.EXAMPLE
    .\run-framework.ps1 -Step 1
    Run only step 1

.EXAMPLE
    .\run-framework.ps1 -Step 4.6
    Run only step 4.6

.EXAMPLE
    .\run-framework.ps1 -From 1 -To 3
    Run steps 1 through 3

.EXAMPLE
    .\run-framework.ps1 -List
    List all available steps
#>

param(
    [string]$Step,
    [string]$From,
    [string]$To,
    [switch]$All,
    [switch]$List
)

# Get the directory where this script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Project root is two levels up from script location (docs/claude-code-agents-framework -> root)
$ProjectRoot = (Get-Item "$ScriptDir\..\..").FullName

# Framework prompts directory relative to project root
$FrameworkDir = "docs\claude-code-agents-framework"

# Define steps in order
$Steps = @(
    "01-step1-find-code-patterns.md",
    "02-step2-teach-patterns-to-ai.md",
    "03-step3-create-ai-assistants.md",
    "04-step4-connect-everything.md",
    "04.5-step4.5-wire-agents-to-skills.md",
    "04.6-step4.6-register-in-claude-md.md",
    "05-step5-test-on-real-work.md",
    "06-step6-automatic-improvements.md",
    "07-weekly-check-system-health.md",
    "08-monthly-clean-up-duplicates.md"
)

function Write-Banner {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Blue
    Write-Host "  Claude Code Agents Framework Runner" -ForegroundColor Blue
    Write-Host "==============================================" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Project root: " -NoNewline
    Write-Host $ProjectRoot -ForegroundColor Green
    Write-Host "Prompts dir:  " -NoNewline
    Write-Host $FrameworkDir -ForegroundColor Green
    Write-Host ""
}

function Write-StepHeader {
    param([int]$StepNum, [string]$StepFile)

    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host "  Step $StepNum`: $StepFile" -ForegroundColor Green
    Write-Host "==============================================" -ForegroundColor Green
    Write-Host ""
}

function Test-ProjectRoot {
    $claudeMdPath = Join-Path $ProjectRoot "CLAUDE.md"

    if (-not (Test-Path $claudeMdPath)) {
        Write-Host "Error: CLAUDE.md not found at project root: $ProjectRoot" -ForegroundColor Red
        Write-Host "Make sure this script is located in docs\claude-code-agents-framework\"
        exit 1
    }

    $claudeDir = Join-Path $ProjectRoot ".claude"
    if (-not (Test-Path $claudeDir)) {
        Write-Host "Note: .claude folder not found yet at project root" -ForegroundColor Yellow
        Write-Host "The framework will create it during step execution."
        Write-Host ""
    }
}

function Invoke-Step {
    param([string]$StepFile)

    $StepPath = Join-Path $ProjectRoot $FrameworkDir $StepFile

    if (-not (Test-Path $StepPath)) {
        Write-Host "Error: File not found: $StepPath" -ForegroundColor Red
        return $false
    }

    Write-Host "Changing to project root: $ProjectRoot" -ForegroundColor Yellow
    Push-Location $ProjectRoot

    try {
        Write-Host "Sending prompt to Claude..." -ForegroundColor Yellow
        Write-Host ""

        # Read file content and pipe to claude (running from project root)
        $content = Get-Content -Path $StepPath -Raw
        $content | claude -p

        Write-Host ""
        Write-Host "Step completed." -ForegroundColor Green
    }
    finally {
        Pop-Location
    }

    return $true
}

function Get-StepIndex {
    param([string]$StepNum)

    for ($i = 0; $i -lt $Steps.Count; $i++) {
        $step = $Steps[$i]
        # Match step number patterns like "01", "4.5", "4.6"
        if ($step -match "^0?$([regex]::Escape($StepNum))[-.]" -or
            $step -match "^$([regex]::Escape($StepNum))-step") {
            return $i
        }
    }

    return -1
}

function Show-Steps {
    Write-Host "Available steps:" -ForegroundColor Cyan
    Write-Host ""

    for ($i = 0; $i -lt $Steps.Count; $i++) {
        Write-Host "  $($i + 1). $($Steps[$i])"
    }
    Write-Host ""
    Write-Host "Use step identifiers: 1, 2, 3, 4, 4.5, 4.6, 5, 6, 7, 8"
    Write-Host ""
}

function Show-Usage {
    Write-Host "Usage:" -ForegroundColor Cyan
    Write-Host "  .\run-framework.ps1                    Run all steps interactively"
    Write-Host "  .\run-framework.ps1 -All               Run all steps without pausing"
    Write-Host "  .\run-framework.ps1 -List              List all available steps"
    Write-Host "  .\run-framework.ps1 -Step <n>          Run a specific step"
    Write-Host "  .\run-framework.ps1 -From <n> -To <m>  Run steps from n to m"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run-framework.ps1 -Step 1            Run step 1 only"
    Write-Host "  .\run-framework.ps1 -Step 4.6          Run step 4.6 only"
    Write-Host "  .\run-framework.ps1 -From 1 -To 3      Run steps 1, 2, and 3"
    Write-Host "  .\run-framework.ps1 -From 4 -To 4.6    Run steps 4, 4.5, and 4.6"
    Write-Host ""
    Write-Host "Note: Claude CLI runs from project root: $ProjectRoot"
    Write-Host ""
}

# Main execution
Write-Banner
Test-ProjectRoot

# Handle -List parameter
if ($List) {
    Show-Steps
    exit 0
}

# Handle -Step parameter (single step)
if ($Step) {
    $stepIdx = Get-StepIndex $Step

    if ($stepIdx -eq -1) {
        Write-Host "Error: Step '$Step' not found" -ForegroundColor Red
        Write-Host ""
        Show-Steps
        exit 1
    }

    Write-StepHeader ($stepIdx + 1) $Steps[$stepIdx]
    $result = Invoke-Step $Steps[$stepIdx]

    if ($result) {
        Write-Host "Step completed!" -ForegroundColor Green
    }
    exit 0
}

# Handle -From and -To parameters (range)
if ($From -and $To) {
    $startIdx = Get-StepIndex $From
    $endIdx = Get-StepIndex $To

    if ($startIdx -eq -1) {
        Write-Host "Error: Step '$From' not found" -ForegroundColor Red
        exit 1
    }

    if ($endIdx -eq -1) {
        Write-Host "Error: Step '$To' not found" -ForegroundColor Red
        exit 1
    }

    for ($i = $startIdx; $i -le $endIdx; $i++) {
        Write-StepHeader ($i + 1) $Steps[$i]
        Invoke-Step $Steps[$i] | Out-Null
        Write-Host ""
    }

    Write-Host "Selected steps completed!" -ForegroundColor Green
    exit 0
}

# Handle -All parameter or default interactive mode
if ($All) {
    # Run all steps without pausing
    for ($i = 0; $i -lt $Steps.Count; $i++) {
        Write-StepHeader ($i + 1) $Steps[$i]
        Invoke-Step $Steps[$i] | Out-Null
        Write-Host ""
    }

    Write-Host "All steps completed!" -ForegroundColor Green
}
else {
    # Interactive mode - pause between steps
    for ($i = 0; $i -lt $Steps.Count; $i++) {
        Write-StepHeader ($i + 1) $Steps[$i]
        Invoke-Step $Steps[$i] | Out-Null
        Write-Host ""

        # Don't pause after the last step
        if ($i -lt ($Steps.Count - 1)) {
            Write-Host "Press Enter to continue to next step (or Ctrl+C to stop)..." -ForegroundColor Yellow
            Read-Host
        }
    }

    Write-Host "All steps completed!" -ForegroundColor Green
}
