@echo off
REM Flask 서버 시작 스크립트

REM 가상환경 활성화 (있는 경우)
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM 서버 시작
echo Starting MOPJ Prediction Server...
python run.py %*