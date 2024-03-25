from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Moderations(BaseModel):
    hap_input: str = 'true'
    threshold: float = 0.75
    hap_output: str = 'true'

class Parameters(BaseModel):
    decoding_method: str = "greedy"
    min_new_tokens: int = 1
    max_new_tokens: int = 500
    repetition_penalty: float = 1.1
    temperature: float = 0.7
    top_k: int = 50
    top_p: int = 1
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

class queryWDLLMRequest(BaseModel):
    question: str
    project_id: str
    collection_id: str
    wd_version: Optional[str] = Field(default='2020-08-30')
    wd_return_params: Optional[List[str]] = Field(default=["Title", "Text"], description="Params to pull from WD. Defaults Title and Text.")
    llm_instructions: Optional[str] = Field(None, title="LLM Instructions", description="Instructions for LLM")
    num_results: Optional[str] = Field(default="5")
    llm_params: Optional[LLMParams] = LLMParams()
    wd_document_names: Optional[List[str]] = Field(None,
        example=["acme.pdf", "test.docx"]
    )

