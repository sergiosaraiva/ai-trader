@echo off
REM =============================================================================
REM Claude Code Agents Framework Runner (Windows Batch)
REM =============================================================================
REM Usage:
REM   run-framework.bat           Run all steps interactively
REM   run-framework.bat --all     Run all steps without pausing
REM   run-framework.bat --list    List all available steps
REM   run-framework.bat 1         Run only step 1
REM   run-framework.bat 4.6       Run only step 4.6
REM
REM Note: Claude CLI runs from the project root where CLAUDE.md and .claude/ exist
REM =============================================================================

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM Project root is two levels up from script location (docs\claude-code-agents-framework -> root)
pushd "%SCRIPT_DIR%\..\.."
set "PROJECT_ROOT=%CD%"
popd

REM Framework prompts directory relative to project root
set "FRAMEWORK_DIR=docs\claude-code-agents-framework"

REM Define steps (using numbered variables since batch doesn't have arrays)
set "STEP_1=01-step1-find-code-patterns.md"
set "STEP_2=02-step2-teach-patterns-to-ai.md"
set "STEP_3=03-step3-create-ai-assistants.md"
set "STEP_4=04-step4-connect-everything.md"
set "STEP_5=04.5-step4.5-wire-agents-to-skills.md"
set "STEP_6=04.6-step4.6-register-in-claude-md.md"
set "STEP_7=05-step5-test-on-real-work.md"
set "STEP_8=06-step6-automatic-improvements.md"
set "STEP_9=07-weekly-check-system-health.md"
set "STEP_10=08-monthly-clean-up-duplicates.md"
set "TOTAL_STEPS=10"

REM Print banner
echo.
echo ==============================================
echo   Claude Code Agents Framework Runner
echo ==============================================
echo.
echo Project root: %PROJECT_ROOT%
echo Prompts dir:  %FRAMEWORK_DIR%
echo.

REM Verify project root
call :verify_project_root

REM Parse arguments
if "%~1"=="" goto :interactive
if "%~1"=="--help" goto :usage
if "%~1"=="-h" goto :usage
if "%~1"=="--list" goto :list
if "%~1"=="-l" goto :list
if "%~1"=="--all" goto :runall
if "%~1"=="-a" goto :runall

REM Run specific step
call :run_single_step %~1
goto :end

:verify_project_root
if not exist "%PROJECT_ROOT%\CLAUDE.md" (
    echo Error: CLAUDE.md not found at project root: %PROJECT_ROOT%
    echo Make sure this script is located in docs\claude-code-agents-framework\
    exit /b 1
)
if not exist "%PROJECT_ROOT%\.claude" (
    echo Note: .claude folder not found yet at project root
    echo The framework will create it during step execution.
    echo.
)
goto :eof

:usage
echo Usage:
echo   %~nx0                  Run all steps interactively
echo   %~nx0 --all            Run all steps without pausing
echo   %~nx0 --list           List all available steps
echo   %~nx0 ^<step^>           Run a specific step (1, 2, 3, 4, 4.5, 4.6, 5, 6, 7, 8)
echo.
echo Examples:
echo   %~nx0 1                Run step 1 only
echo   %~nx0 4.6              Run step 4.6 only
echo   %~nx0 --all            Run all steps automatically
echo.
echo Note: Claude CLI runs from project root: %PROJECT_ROOT%
goto :end

:list
echo Available steps:
echo.
echo   1.  %STEP_1%
echo   2.  %STEP_2%
echo   3.  %STEP_3%
echo   4.  %STEP_4%
echo   5.  %STEP_5% (Step 4.5)
echo   6.  %STEP_6% (Step 4.6)
echo   7.  %STEP_7%
echo   8.  %STEP_8%
echo   9.  %STEP_9%
echo   10. %STEP_10%
echo.
echo Use step numbers 1-10, or special: 4.5, 4.6
goto :end

:runall
echo Running all steps without pausing...
echo.
for /L %%i in (1,1,%TOTAL_STEPS%) do (
    call :run_step %%i 0
)
echo.
echo All steps completed!
goto :end

:interactive
echo Running all steps interactively...
echo.
for /L %%i in (1,1,%TOTAL_STEPS%) do (
    call :run_step %%i 1
)
echo.
echo All steps completed!
goto :end

:run_single_step
set "REQUESTED=%~1"

REM Map input to step number
if "%REQUESTED%"=="1" set "STEP_NUM=1" & goto :do_single
if "%REQUESTED%"=="2" set "STEP_NUM=2" & goto :do_single
if "%REQUESTED%"=="3" set "STEP_NUM=3" & goto :do_single
if "%REQUESTED%"=="4" set "STEP_NUM=4" & goto :do_single
if "%REQUESTED%"=="4.5" set "STEP_NUM=5" & goto :do_single
if "%REQUESTED%"=="4.6" set "STEP_NUM=6" & goto :do_single
if "%REQUESTED%"=="5" set "STEP_NUM=7" & goto :do_single
if "%REQUESTED%"=="6" set "STEP_NUM=8" & goto :do_single
if "%REQUESTED%"=="7" set "STEP_NUM=9" & goto :do_single
if "%REQUESTED%"=="8" set "STEP_NUM=10" & goto :do_single

echo Error: Step '%REQUESTED%' not found
echo.
goto :list

:do_single
call :run_step %STEP_NUM% 0
echo.
echo Step completed!
goto :eof

:run_step
set "NUM=%~1"
set "PAUSE_AFTER=%~2"

REM Get the step file name
set "STEP_FILE=!STEP_%NUM%!"

if not defined STEP_FILE (
    echo Error: Invalid step number %NUM%
    goto :eof
)

set "STEP_PATH=%PROJECT_ROOT%\%FRAMEWORK_DIR%\%STEP_FILE%"

if not exist "%STEP_PATH%" (
    echo Error: File not found: %STEP_PATH%
    goto :eof
)

echo ==============================================
echo   Step %NUM%: %STEP_FILE%
echo ==============================================
echo.
echo Changing to project root: %PROJECT_ROOT%
pushd "%PROJECT_ROOT%"

echo Sending prompt to Claude...
echo.

REM Pipe the file content to claude (running from project root)
type "%STEP_PATH%" | claude -p

popd

echo.
echo Step %NUM% completed.
echo.

REM Pause if in interactive mode and not the last step
if "%PAUSE_AFTER%"=="1" (
    if not "%NUM%"=="%TOTAL_STEPS%" (
        echo Press any key to continue to next step ^(or Ctrl+C to stop^)...
        pause >nul
        echo.
    )
)

goto :eof

:end
endlocal
