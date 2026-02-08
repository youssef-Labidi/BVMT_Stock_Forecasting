# BVMT Forecasting API — Run Instructions

Prerequisites:
- Python 3.10+ recommended

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the API locally:

```bash
# from project root
python api_server.py
# or with uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Test the endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"symbol":"BIAT","forecast_days":5}'
```

Notes:
- The server uses the `prophet` implementation inside `bvmt_forecasting_module.py` (it is the best-performing model per module summary).
- For production, connect `BVMTDataPipeline.load_data` to real data and secure the API.
