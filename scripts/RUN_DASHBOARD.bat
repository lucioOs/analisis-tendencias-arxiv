@echo off
setlocal EnableExtensions
chcp 65001 >nul

REM Ir a la raiz del proyecto (este .bat esta en /scripts)
cd /d "%~dp0.."

REM Validar venv
if not exist "venv\Scripts\python.exe" (
  echo ERROR: No se encontro venv\Scripts\python.exe
  echo Crea el venv o revisa la carpeta venv.
  pause
  exit /b 1
)

REM Activar venv
call "venv\Scripts\activate.bat"

REM Verificar streamlit instalado
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
  echo Instalando dependencias...
  if exist "requirements.txt" (
    python -m pip install -r requirements.txt
  ) else (
    echo ERROR: No existe requirements.txt
    pause
    exit /b 1
  )
)

REM Verificar que exista la app
if not exist "app\streamlit_app.py" (
  echo ERROR: No se encontro app\streamlit_app.py
  pause
  exit /b 1
)

REM Lanzar dashboard
echo Iniciando dashboard en http://localhost:8501 ...
python -m streamlit run "app\streamlit_app.py" --server.port 8501 --server.address 127.0.0.1

pause
endlocal

