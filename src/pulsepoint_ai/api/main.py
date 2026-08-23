import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pulsepoint_ai.api.routers import connect, predict, triage

logger = structlog.get_logger()

app = FastAPI(
    title="PulsePoint AI Engine",
    description="AI/ML Microservice for the PulsePoint Health Network",
    version="3.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(triage.router, prefix="/api/v1/triage", tags=["Triage"])
app.include_router(predict.router, prefix="/api/v1/predict", tags=["Predict"])
app.include_router(connect.router, prefix="/api/v1/connect", tags=["Connect"])

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "pulsepoint-ai",
        "version": "3.0.0",
        "engines": ["triage", "predict", "connect"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
