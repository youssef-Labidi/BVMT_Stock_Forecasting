@echo off
REM push_subfolder.bat <remote.git.url> <subfolder> <branch>
REM Example: push_subfolder.bat https://github.com/Homssalomssa/bvmt-sentiment-analysis.git bvmt-forecasting add-bvmt-forecasting

if "%1"=="" (
  echo Usage: push_subfolder.bat ^<remote.git.url^> ^<subfolder^> ^<branch^>
  exit /b 1
)





















echo Done. Create a PR from branch %BRANCH% in the target repo.
ngit add .
ngit commit -m "Move project into subfolder %SUBFOLDER%"
ngit remote add target %REMOTE% 2>nul || echo remote 'target' already exists
ngit push target %BRANCH%)  )    )      git mv "%%F" "%SUBFOLDER%/%%F" >nul 2>&1 || echo Skipped: %%F      for %%D in ("%%~dpF") do if not exist "%SUBFOLDER%\%%~pF" mkdir "%SUBFOLDER%\%%~pF" 2>nul      echo Moving %%F -> %SUBFOLDER%/%%F    if NOT "%%F"=="%SUBFOLDER%\%%F" (  if NOT "%%F"=="%~nx0" (
nREM Move tracked files into subfolder
nfor /f "usebackq delims=" %%F in (`git ls-files`) do (
nREM Create subfolder
nif not exist "%SUBFOLDER%" mkdir "%SUBFOLDER%"
nREM create branch
ngit checkout -b %BRANCH%)  exit /b 1  echo Working tree not clean. Please commit or stash changes first.if "%BRANCH%"=="" set BRANCH=add-bvmt-forecasting
n
nREM Ensure clean working tree
ngit status --porcelain >nul 2>&1
nif not errorlevel 1 (set BRANCH=%3set SUBFOLDER=%2nset REMOTE=%1