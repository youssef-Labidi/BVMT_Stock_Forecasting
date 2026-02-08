from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from bvmt_forecasting_module import predict_stock

app = FastAPI(title="BVMT Forecasting API")

class PredictRequest(BaseModel):
    symbol: str
    forecast_days: Optional[int] = 5

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(req: PredictRequest):
    symbol = req.symbol
    days = req.forecast_days or 5
    try:
        result = predict_stock(symbol, model_type='prophet', forecast_days=days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, log_level="info")
