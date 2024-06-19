from pydantic import BaseModel, Field
from typing import Optional, List

class texttosqlRequest(BaseModel):
    nl: str = Field(title="NL Question", description="Question asked by the user.")