from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SQLQueryResponse(BaseModel):
    detail: str = Field(..., description="Detail message")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="List of query results")
