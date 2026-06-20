from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import CustomerInput, SegmentOutput
from app.model import predict_segment

app = FastAPI(
    title="Customer Segmentation API",
    description="Predicts customer segment based on RFM features using K-Means clustering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "kmeans_k4"}

@app.post("/predict", response_model=SegmentOutput)
def predict(customer: CustomerInput):
    result = predict_segment(
        recency=customer.recency,
        frequency=customer.frequency,
        monetary=customer.monetary,
        avg_order_value=customer.avg_order_value,
        customer_lifetime=customer.customer_lifetime
    )

    return SegmentOutput(
        segment=result["segment"],
        cluster_id=result["cluster_id"],
        recency=customer.recency,
        frequency=customer.frequency,
        monetary=customer.monetary,
        strategy=result["strategy"],
        top_action=result["top_action"]
    )