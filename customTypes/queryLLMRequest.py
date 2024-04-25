from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class Moderations(BaseModel):
    hap_input: str = 'true'
    threshold: float = 0.75
    hap_output: str = 'true'

class Parameters(BaseModel):
    decoding_method: str = "greedy"
    min_new_tokens: int = 1
    max_new_tokens: int = 500
    moderations: Moderations = Moderations()

    def dict(self, *args, **kwargs):
        """
        Override dict() method to return a dictionary representation
        """
        params_dict = super().dict(*args, **kwargs)
        params_dict['moderations'] = self.moderations.dict()
        return params_dict

class LLMParams(BaseModel):
    model_id: str = "meta-llama/llama-2-70b-chat"
    inputs: list = []
    parameters: Parameters = Parameters()

    # Resolves warning error with model_id:
    #     Field "model_id" has conflict with protected namespace "model_".
    #     You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    #     warnings.warn(
    class Config:
        protected_namespaces = ()

class queryLLMRequest(BaseModel):
    llm_instructions: Optional[str] = Field(None, title="LLM Instructions", description="Instructions for LLM")
    question: str
    es_index_name: str
    es_index_text_field: Optional[str] = Field(default="body_content_field")
    es_model_name: Optional[str] = Field(default=".elser_model_1")
    num_results: Optional[str] = Field(default="5")
    llm_params: Optional[LLMParams] = LLMParams()
    filters: Optional[Dict[str, Any]] = Field(None,
        example={
            "date": "2022-01-01",
            "file_name": "test.pdf"
        })

