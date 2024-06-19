from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class texttosqlResponse(BaseModel):
     results: List[Dict[str, Any]] = Field(None, description="List of query results")