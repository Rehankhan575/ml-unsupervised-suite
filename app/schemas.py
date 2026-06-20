from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    recency: float = Field(..., description="Days since last purchase", gt=0)
    frequency: float = Field(..., description="Number of unique invoices", gt=0)
    monetary: float = Field(..., description="Total spend in GBP", gt=0)
    avg_order_value: float = Field(..., description="Average spend per invoice", gt=0)
    customer_lifetime: float = Field(..., description="Days between first and last purchase", ge=0)


class SegmentOutput(BaseModel):
    segment: str
    cluster_id: int
    recency: float
    frequency: float
    monetary: float
    strategy: str
    top_action: str