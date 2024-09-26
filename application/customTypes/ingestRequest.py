from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ingestRequest(BaseModel):
    #bucket_name: str = Field(title="COS Bucket Name", description="Name of your cloud object storage bucket.")
    GUID: str = Field(title="GUID", description="Unique identifier for the document.")
    title: str = Field(title="Title", description="Title of the document.")
    URL: str = Field(title="Document URL", description="URL where the document is hosted.")
    content: str = Field(title="Content", description="Main body content of the document.")
    tags: List[str] = Field(title="Tags", description="List of tags associated with the document.")
    updated_date: datetime = Field(title="Update Date", description="Date when the document was last updated.")
    view_security_roles: List[str] = Field(title="View Security Roles", description="List of security roles")
    es_index_name: str = Field(title="ElasticSearch Index Name", description="Name of the elasticsearch index you want to create.")
    es_pipeline_name: str = Field(title="ElasticSearch Pipeline Name", description="Name of the elasticsearch pipeline you want to create.")
    chunk_size: Optional[str] = Field(default="512")
    chunk_overlap: Optional[str] = Field(default="256")
    es_model_name: Optional[str] = Field(default=".elser_model_2_linux-x86_64")
    es_model_text_field: Optional[str] = Field(default="text_field") 
    es_index_text_field: Optional[str] = Field(default="content")
    # TODO: Implement metadata
    # metadata_fields: Optional[List[str]] = None
