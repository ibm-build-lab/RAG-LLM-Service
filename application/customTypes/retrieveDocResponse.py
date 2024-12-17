from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class retrieveDocResponse(BaseModel):
    references: List[Dict]
    # file_name: Optional[str]
    # url: Optional[str]
    # text: str
    #llama_document: Optional[dict] = None