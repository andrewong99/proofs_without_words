@echo off
rem SPDX-License-Identifier: GPL-3.0-or-later
setlocal EnableExtensions DisableDelayedExpansion
rem ===========================================================================
rem  pww.bat  --  double-click launcher for Proofs Without Words (pww_app.py)
rem
rem  Starts the app with a Python that has its packages, through that Python's
rem  pythonw.exe, so no console window stays open behind it.
rem
rem  First run: if no Python here has the packages, it sets them up inside
rem  this folder, in .runtime, after a 30-second countdown you can cancel:
rem      1. uv, a Python installer, from PyPI (its SHA-256 is checked)
rem      2. Python 3.12: the one on this computer if there is one, otherwise
rem         uv downloads it into .runtime
rem      3. a virtual environment, .runtime\venv, with requirements.txt
rem  PATH, the registry and any other Python are left alone; deleting
rem  .runtime undoes it all. Later, if requirements.txt changes, the next
rem  start brings .runtime\venv up to date. Set PWW_NO_SETUP=1 to switch the
rem  setup off.
rem
rem  Search order for a Python that has PyQt6, matplotlib, manim and vpython:
rem      1. PWW_PYTHON, if set (on the line below, or as an environment var)
rem      2. .venv\Scripts\python.exe next to this file
rem      3. .runtime\venv\Scripts\python.exe (made by the setup)
rem      4. python on PATH
rem      5. the py launcher: py -3.12, py -3.13, then py -3
rem      6. %LOCALAPPDATA%\Programs\Python\Python312
rem
rem  To force a particular Python, delete "rem " at the start of the next
rem  line and put the full path to its python.exe:
rem set "PWW_PYTHON=C:\full\path\to\python.exe"
rem ===========================================================================

set "HERE=%~dp0"
set "APP=%~dp0pww_app.py"
set "REQ=%~dp0requirements.txt"
set "RUNTIME=%~dp0.runtime"
set "ENVDIR=%~dp0.runtime\venv"
set "ENVPY=%~dp0.runtime\venv\Scripts\python.exe"
set "UV=%~dp0.runtime\uv.exe"
rem  Windows' own tools, by full path, so that a GNU tar or timeout that came
rem  with Git or MSYS2 and sits earlier on PATH is never picked up instead.
set "SYS=%SystemRoot%\System32"

rem  uv, pinned: its Windows x64 wheel on PyPI, and that file's SHA-256
set "UV_VER=0.12.19"
set "UV_URL=https://files.pythonhosted.org/packages/c2/5d/8e0b84503b77ead843ef57e4f9305eb32a95cac6806c0e0d54b9404a5f7b/uv-0.12.19-py3-none-win_amd64.whl"
set "UV_SHA256=dcbc531a96762569bbfe9639b4f45f00aabff51f427540711f63e7c23f225fdf"
set "UV_MEMBER=uv-0.12.19.data/scripts/uv.exe"

cd /d "%HERE%"
if not exist "%APP%" goto :no_app

call :find
if not defined PY_FULL goto :not_full
if /i not "%PY_FULL%"=="%ENVPY%" goto :launch
call :sync_env
:launch
call :start_app "%PY_FULL%"
exit /b 0

:not_full
set "WHY=nosetup"
if defined PWW_NO_SETUP goto :not_ready
set "WHY=forced"
if defined PWW_PYTHON goto :not_ready
goto :setup


rem ---------------------------------------------------------------------------
:setup
title Proofs Without Words - first-time setup
echo.
echo  Proofs Without Words: first-time setup
echo  ======================================
echo.
echo  No Python on this computer has everything pww needs yet. The setup puts
echo  it all inside this folder, in .runtime:
echo.
echo    1. uv %UV_VER%, a Python installer (18 MB from PyPI, SHA-256 checked)
echo    2. Python 3.12: the one on this computer if there is one, otherwise
echo       uv downloads it (about 25 MB)
echo    3. the packages in requirements.txt (about 400 MB to download)
echo.
echo  It takes a few minutes and about 1.5 GB of disk. PATH, the registry and
echo  any other Python are left alone. Deleting .runtime undoes it all.
echo.
"%SYS%\choice.exe" /c YN /t 30 /d Y /m " Set up now? It starts by itself in 30 seconds"
if errorlevel 2 goto :setup_declined
if not errorlevel 1 goto :setup_declined

call :need_tools
if errorlevel 1 goto :setup_failed
call :uv_env
call :get_uv
if errorlevel 1 goto :setup_failed
call :make_env
if errorlevel 1 goto :setup_failed
call :install_packages
if errorlevel 1 goto :setup_failed
copy /y "%REQ%" "%RUNTIME%\requirements.installed" >nul
call :tidy

call :find
if not defined PY_FULL goto :setup_unfinished
echo.
echo  Setup finished. Starting Proofs Without Words ...
call :start_app "%PY_FULL%"
exit /b 0

:setup_declined
set "WHY=declined"
goto :not_ready

:setup_unfinished
set "FAIL=the packages are installed, but the app's check still fails with .runtime\venv."
goto :setup_failed

:setup_failed
echo.
echo  The setup stopped: %FAIL%
echo.
echo  Run pww.bat again to retry; it carries on from where it stopped.
echo  Everything it made is in .runtime, which is safe to delete.
echo.
if defined PY_GUI goto :setup_failed_gui
pause
exit /b 1

:setup_failed_gui
echo  Press any key to open the catalogue without rendering, or close this
echo  window.
pause >nul
call :start_app "%PY_GUI%"
exit /b 0


rem ---------------------------------------------------------------------------
:not_ready
echo.
if "%WHY%"=="declined" echo  Setup skipped. Run pww.bat again whenever you want it.
if "%WHY%"=="nosetup" echo  No Python here has the app's packages, and the setup is off: PWW_NO_SETUP is set.
if "%WHY%"=="forced" echo  PWW_PYTHON is set, so pww.bat uses that Python only and sets nothing up.
set "HOWPY=python"
if defined PY_GUI set "HOWPY=%PY_GUI%"
echo  To install the app's packages into a Python yourself:
echo.
echo      "%HOWPY%" -m pip install -r "%REQ%"
echo.
if defined PY_GUI goto :not_ready_gui
echo  No Python with PyQt6 and matplotlib was found, so the catalogue cannot
echo  open yet.
echo.
pause
exit /b 1

:not_ready_gui
echo  This Python can open the catalogue, but not render proofs:
echo      "%PY_GUI%"
echo.
echo  Opening the catalogue in 30 seconds (any key opens it now).
"%SYS%\timeout.exe" /t 30 >nul
call :start_app "%PY_GUI%"
exit /b 0


:no_app
echo.
echo  pww_app.py is not in the same folder as this file:
echo      "%HERE%"
echo.
echo  Keep pww.bat beside pww_app.py. To start it from the desktop, make a
echo  shortcut instead: right-click pww.bat, Send to, Desktop.
echo.
pause
exit /b 1


rem ---------------------------------------------------------------------------
rem  Setup steps. Each ends with "exit /b 0", or sets FAIL and ends with
rem  "exit /b 1".

:need_tools
"%SYS%\curl.exe" --version >nul 2>&1
if %errorlevel% neq 0 goto :need_tools_missing
"%SYS%\tar.exe" --version >nul 2>&1
if %errorlevel% neq 0 goto :need_tools_missing
exit /b 0
:need_tools_missing
set "FAIL=it needs curl.exe and tar.exe, which come with Windows 10 (version 1803 or newer) and Windows 11."
exit /b 1


:uv_env
rem  Keep everything uv makes inside .runtime.
set "UV_CACHE_DIR=%RUNTIME%\cache"
set "UV_PYTHON_INSTALL_DIR=%RUNTIME%\python"
set "UV_PYTHON_BIN_DIR=%RUNTIME%\bin"
set "UV_PYTHON_INSTALL_BIN=0"
set "UV_PYTHON_INSTALL_REGISTRY=0"
set "UV_PYTHON_DOWNLOADS=automatic"
exit /b 0


:get_uv
if not exist "%UV%" goto :get_uv_download
"%UV%" --version 2>nul | "%SYS%\findstr.exe" /b /c:"uv %UV_VER% " >nul
if %errorlevel% equ 0 exit /b 0
del "%UV%" >nul 2>&1
:get_uv_download
if not exist "%RUNTIME%" mkdir "%RUNTIME%" 2>nul
if not exist "%RUNTIME%" goto :get_uv_nofolder
set "UV_WHL=%RUNTIME%\uv-%UV_VER%.whl"
echo.
echo  [1/3] Downloading uv %UV_VER% ...
"%SYS%\curl.exe" -fL --retry 3 --retry-delay 2 -# -o "%UV_WHL%" "%UV_URL%"
if %errorlevel% neq 0 goto :get_uv_nodownload
"%SYS%\certutil.exe" -hashfile "%UV_WHL%" SHA256 | "%SYS%\findstr.exe" /i /c:"%UV_SHA256%" >nul
if %errorlevel% neq 0 goto :get_uv_badhash
"%SYS%\tar.exe" -xf "%UV_WHL%" --strip-components=2 -C "%RUNTIME%" "%UV_MEMBER%"
if %errorlevel% neq 0 goto :get_uv_unpack
if not exist "%UV%" goto :get_uv_unpack
del "%UV_WHL%" >nul 2>&1
"%UV%" --version
if %errorlevel% neq 0 goto :get_uv_broken
exit /b 0
:get_uv_nofolder
set "FAIL=it cannot create .runtime here, because this folder is not writable. Move the pww folder somewhere you can write to, such as Documents."
exit /b 1
:get_uv_nodownload
del "%UV_WHL%" >nul 2>&1
set "FAIL=could not download uv from PyPI (files.pythonhosted.org). Check the internet connection or proxy."
exit /b 1
:get_uv_badhash
del "%UV_WHL%" >nul 2>&1
set "FAIL=the uv download did not match its SHA-256, so it was deleted."
exit /b 1
:get_uv_unpack
set "FAIL=could not unpack uv.exe from the download."
exit /b 1
:get_uv_broken
set "FAIL=.runtime\uv.exe would not start; an antivirus may have stopped it."
exit /b 1


:make_env
echo.
echo  [2/3] Python 3.12 and a virtual environment for pww ...
if not exist "%ENVPY%" goto :make_env_new
"%ENVPY%" -c "import sys" >nul 2>&1
if %errorlevel% equ 0 exit /b 0
:make_env_new
"%UV%" venv "%ENVDIR%" --python cpython-3.12-windows-x86_64-none --seed --clear --force
if %errorlevel% neq 0 goto :make_env_failed
if not exist "%ENVPY%" goto :make_env_failed
exit /b 0
:make_env_failed
set "FAIL=uv could not make the Python 3.12 environment (its message is above). When this computer has no Python 3.12, uv downloads one from GitHub."
exit /b 1


:install_packages
echo.
echo  [3/3] Installing the packages in requirements.txt ...
"%UV%" pip install --python "%ENVPY%" -r "%REQ%"
if %errorlevel% neq 0 goto :install_failed
exit /b 0
:install_failed
set "FAIL=uv could not install the packages (its message is above)."
exit /b 1


:tidy
rem  Empty uv's download cache. uv does it itself: the cache holds paths longer
rem  than the 260 characters that rmdir can handle.
"%UV%" cache clean >nul 2>&1
if exist "%RUNTIME%\cache" rmdir /s /q "%RUNTIME%\cache" >nul 2>&1
exit /b 0


:sync_env
rem  .runtime\venv is in use: if requirements.txt has changed since it was
rem  set up, install what is new. Starts the app anyway if that fails.
if not exist "%UV%" exit /b 0
"%SYS%\fc.exe" /b "%REQ%" "%RUNTIME%\requirements.installed" >nul 2>&1
if %errorlevel% equ 0 exit /b 0
echo.
echo  requirements.txt has changed since the setup; updating .runtime\venv ...
call :uv_env
"%UV%" pip install --python "%ENVPY%" -r "%REQ%"
if %errorlevel% neq 0 goto :sync_env_failed
copy /y "%REQ%" "%RUNTIME%\requirements.installed" >nul
call :tidy
exit /b 0
:sync_env_failed
echo.
echo  The update did not finish (see above); starting the app as it is.
"%SYS%\timeout.exe" /t 10 >nul
exit /b 0


rem ---------------------------------------------------------------------------
:find
rem  Sets PY_FULL (can render) and PY_GUI (can at least open the window).
set "PY_FULL="
set "PY_GUI="
if not defined PWW_PYTHON goto :find_rest
call :consider "%PWW_PYTHON%"
:find_rest
call :consider "%HERE%.venv\Scripts\python.exe"
call :consider "%ENVPY%"
call :consider_cmd python
call :consider_cmd py -3.12
call :consider_cmd py -3.13
call :consider_cmd py -3
call :consider "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
exit /b 0


rem ---------------------------------------------------------------------------
:start_app
rem  %1 = python.exe. Prefer the pythonw.exe beside it: no console window.
set "PYW=%~dp1pythonw.exe"
if exist "%PYW%" goto :start_windowless
rem  No pythonw.exe there (unusual): run in this console instead, so that
rem  any error stays on screen.
"%~1" "%APP%"
if %errorlevel% neq 0 pause
goto :eof

:start_windowless
start "" "%PYW%" "%APP%"
goto :eof


rem ---------------------------------------------------------------------------
:consider
rem  %1 = a candidate python.exe. Records the first interpreter that can open
rem  the window (PY_GUI) and the first that can also render (PY_FULL).
rem  Results are tested with "%errorlevel% neq 0", not "if errorlevel 1":
rem  the latter means >= 1 and would accept a crashed interpreter, whose
rem  exit code (e.g. 0xC0000135, DLL not found) is negative.
if defined PY_FULL goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import PyQt6.QtWidgets, matplotlib" >nul 2>&1
if %errorlevel% neq 0 goto :eof
if not defined PY_GUI set "PY_GUI=%~1"
"%~1" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('manim') and u.find_spec('vpython') else 1)" >nul 2>&1
if %errorlevel% neq 0 goto :eof
set "PY_FULL=%~1"
goto :eof


rem ---------------------------------------------------------------------------
:consider_cmd
rem  %* = a command that starts Python ("python", "py -3.12" ...). Resolves it
rem  to the real python.exe, which is also where pythonw.exe lives.
if defined PY_FULL goto :eof
set "CAND="
for /f "usebackq delims=" %%P in (`%* -c "import sys; print(sys.executable)" 2^>nul`) do set "CAND=%%P"
if defined CAND call :consider "%CAND%"
goto :eof
