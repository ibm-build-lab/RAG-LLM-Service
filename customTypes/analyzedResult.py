from pydantic import BaseModel

class analyzedResult(BaseModel):
    severity :str
    important :str