from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class retrieveDocRequest(BaseModel):
    question: str
    es_index_name: str
    es_index_text_field: Optional[str] = Field(default="body_content_field")
    es_model_name: Optional[str] = Field(default=".elser_model_2_linux-x86_64")
    es_model_text_field: Optional[str] = Field(default="ml.tokens")
    num_results: Optional[str] = Field(default="5")
    filters: Optional[Dict[str, Any]] = Field(None,
        example={
            "date": "2022-01-01",
            "file_name": "test.pdf"
        })

