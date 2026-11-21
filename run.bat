@echo off
echo Starting Early Cancer Detection App...
call "%~dp0.venv\Scripts\activate.bat"
python -m streamlit run "%~dp0app.py"
pause