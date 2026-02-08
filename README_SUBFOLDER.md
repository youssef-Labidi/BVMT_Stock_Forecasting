BVMT Forecasting Module

Contents:
- `bvmt_forecasting_module.py` — data pipeline, models (ARIMA, LSTM, Prophet), evaluation, and prediction API functions.
- `api_server.py` — FastAPI server exposing `/predict` endpoint using the Prophet model.
- `requirements.txt` — Python dependencies (includes FastAPI/uvicorn).
- `PUSH_TO_GITHUB.md` — instructions to push this project into a subfolder of a remote repo.
- `push_subfolder.bat` / `push_subfolder.sh` — helper scripts to automate moving tracked files into a subfolder and pushing a branch to a remote.

Notes:
- The forecasting module uses synthetic data by default. Update `BVMTDataPipeline.load_data` to connect to real BVMT data.
- Review `push_subfolder.*` before running — they attempt to move tracked files and create a branch.
