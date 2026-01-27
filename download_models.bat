@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM Qwen3-TTS Model Downloader (for ComfyUI custom_nodes)
REM - Safe for Transformers: pin huggingface_hub < 1.0
REM - Use huggingface-cli.exe (NOT python -m huggingface_hub.cli)
REM ==========================================================

REM --------- 1) Resolve paths ----------
set "NODE_DIR=%~dp0"
if "%NODE_DIR:~-1%"=="\" set "NODE_DIR=%NODE_DIR:~0,-1%"

set "COMFYUI_DIR=%NODE_DIR%\..\.."
for %%I in ("%COMFYUI_DIR%") do set "COMFYUI_DIR=%%~fI"

set "PY=F:\ComfyUI-aki-v1.7\python\python.exe"
set "HFCLI=F:\ComfyUI-aki-v1.7\python\Scripts\hf.exe"
set "MODELS_DIR=F:\ComfyUI-aki-v1.7\models\qwen3_tts"

echo.
echo [INFO] Node dir    : %NODE_DIR%
echo [INFO] ComfyUI dir : %COMFYUI_DIR%
echo [INFO] Python      : %PY%
echo [INFO] HF CLI      : %HFCLI%
echo [INFO] Models dir  : %MODELS_DIR%
echo.

REM --------- 2) Sanity checks ----------
if not exist "%PY%" (
  echo [ERROR] ComfyUI python not found: %PY%
  pause
  exit /b 1
)

if not exist "%HFCLI%" (
  echo [ERROR] huggingface-cli.exe not found: %HFCLI%
  echo         Try: %PY% -m pip install -U "huggingface_hub>=0.34,<1.0"
  pause
  exit /b 1
)

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

REM --------- 3) Pin huggingface_hub to <1.0 (Transformers compatible) ----------
echo [STEP] Ensuring huggingface_hub version (>=0.34,<1.0) ...
"%PY%" -m pip install -U "huggingface_hub>=0.34,<1.0"
if errorlevel 1 (
  echo [ERROR] Failed to install compatible huggingface_hub
  pause
  exit /b 1
)

REM --------- 4) Repos ----------
set "REPO_BASE=Qwen/Qwen3-TTS-12Hz-1.7B-Base"
set "DIR_BASE=%MODELS_DIR%\Qwen3-TTS-12Hz-1.7B-Base"

set "REPO_TOKENIZER=Qwen/Qwen3-TTS-Tokenizer-12Hz"
set "DIR_TOKENIZER=%MODELS_DIR%\Qwen3-TTS-Tokenizer-12Hz"

set "REPO_VOICEDESIGN=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
set "DIR_VOICEDESIGN=%MODELS_DIR%\Qwen3-TTS-12Hz-1.7B-VoiceDesign"

set "REPO_CUSTOMVOICE=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
set "DIR_CUSTOMVOICE=%MODELS_DIR%\Qwen3-TTS-12Hz-1.7B-CustomVoice"

REM --------- 5) Download required ----------
echo.
echo [STEP] Downloading REQUIRED models...
echo.

echo [DL] %REPO_BASE%
"%HFCLI%" download "%REPO_BASE%" --local-dir "%DIR_BASE%"
if errorlevel 1 (
  echo [ERROR] Failed: %REPO_BASE%
  pause
  exit /b 1
)

echo.
echo [DL] %REPO_TOKENIZER%
"%HFCLI%" download "%REPO_TOKENIZER%" --local-dir "%DIR_TOKENIZER%"
if errorlevel 1 (
  echo [ERROR] Failed: %REPO_TOKENIZER%
  pause
  exit /b 1
)

echo.
echo [INFO] Required models downloaded.
echo.

REM --------- 6) Optional ----------
echo Download OPTIONAL models (VoiceDesign / CustomVoice)? (Y/N)
set /p opt=Choice:

if /I "%opt%"=="Y" (
  echo.
  echo [STEP] Downloading OPTIONAL models...
  echo.

  echo [DL] %REPO_VOICEDESIGN%
  "%HFCLI%" download "%REPO_VOICEDESIGN%" --local-dir "%DIR_VOICEDESIGN%"
  if errorlevel 1 (
    echo [ERROR] Failed: %REPO_VOICEDESIGN%
    pause
    exit /b 1
  )

  echo.
  echo [DL] %REPO_CUSTOMVOICE%
  "%HFCLI%" download "%REPO_CUSTOMVOICE%" --local-dir "%DIR_CUSTOMVOICE%"
  if errorlevel 1 (
    echo [ERROR] Failed: %REPO_CUSTOMVOICE%
    pause
    exit /b 1
  )

  echo.
  echo [INFO] Optional models downloaded.
) else (
  echo.
  echo [INFO] Skipped optional downloads.
)

echo.
echo [DONE] All downloads finished.
echo Models location:
echo   %MODELS_DIR%
echo.
pause
endlocal
