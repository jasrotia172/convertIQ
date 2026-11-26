"""API request and response contracts."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    SpecialDay: float = Field(ge=0, le=1)
    Month: Literal["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    OperatingSystems: int = Field(ge=1)
    Browser: int = Field(ge=1)
    Region: int = Field(ge=1)
    TrafficType: int = Field(ge=1)
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"]
    Weekend: bool


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessions: list[SessionInput] = Field(min_length=1, max_length=10000)


class PredictionResponse(BaseModel):
    predicted_class: int
    conversion_probability: float
    propensity_segment: Literal["low", "medium", "high"]
    model_version: str
    decision_threshold: float


class ExplanationResponse(BaseModel):
    method: str
    units: str
    base_value: float | None
    probability: float
    contributions: dict[str, float]
    note: str
