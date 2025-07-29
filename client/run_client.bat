@echo off
echo Starting NaiveHub Streamlit Client...
echo.
echo Make sure all servers are running:
echo - Storage Server: http://localhost:8002
echo - Training Server: http://localhost:8001  
echo - Prediction Server: http://localhost:8003
echo.
pause
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Streamlit client...
streamlit run naivehub_client_v2.py
pause
