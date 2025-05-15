from pydantic import BaseModel, validator
from typing import List

class InputData(BaseModel):
    features: List[float]
    
    @validator('features')
    def validate_features_length(cls, v):
        if len(v) != 4:
            raise ValueError('features must contain exactly 4 elements')
        return v

class Prediction(BaseModel):
    prediction: int