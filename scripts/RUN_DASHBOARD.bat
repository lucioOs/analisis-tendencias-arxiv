@echo on
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM 1) Ir a raíz del proyecto
cd /d "%~dp0.."
if errorlevel 1 (
  echo [ERROR] No pude moverme a la raiz del proyecto.
  pause
  exit /b 1
)

REM 2) Logs
if not exist "logs" mkdir "logs"
set "TS=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TS=%TS: =0%"
set "LOGFILE=logs\dashboard_%TS%.log"
echo [INFO] Log: %LOGFILE%

REM 3) Detectar venv
set "PYEXE="
if exist ".venv\Scripts\python.exe" set "PYEXE=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe"  set "PYEXE=venv\Scripts\python.exe"

if "%PYEXE%"=="" (
  echo [ERROR] No se encontro .venv\Scripts\python.exe ni venv\Scripts\python.exe
  echo [TIP] Crea el venv: python -m venv .venv
  pause
  exit /b 1
)

REM 4) Activar venv
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else (
  call "venv\Scripts\activate.bat"
)

if errorlevel 1 (
  echo [ERROR] No pude activar el venv.
  pause
  exit /b 1
)

echo [OK] venv activado.
python -V

REM 5) Instalar deps si hace falta
python -c "import streamlit" >> "%LOGFILE%" 2>&1
if errorlevel 1 (
  echo [INFO] Instalando requirements.txt...
  python -m pip install -r requirements.txt >> "%LOGFILE%" 2>&1
  if errorlevel 1 (
    echo [ERROR] Fallo instalacion. Revisa: %LOGFILE%
    pause
    exit /b 1
  )
)

REM 6) Verificar app
if not exist "app\streamlit_app.py" (
  echo [ERROR] No existe app\streamlit_app.py
  pause
  exit /b 1
)

REM 7) Abrir navegador y correr
start "" "http://127.0.0.1:8501"
python -m streamlit run "app\streamlit_app.py" --server.port 8501 --server.address 127.0.0.1 >> "%LOGFILE%" 2>&1

echo.
echo [INFO] Streamlit se cerro o fallo. Log: %LOGFILE%
pause
endlocal
