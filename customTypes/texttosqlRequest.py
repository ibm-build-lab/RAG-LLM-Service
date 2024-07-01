from pydantic import BaseModel, Field
from typing import Optional, List

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

class LLMParams(BaseModel):
    model_id: str = "meta-llama/llama-3-70b-instruct"
    inputs: list = []
    parameters: Parameters = Parameters()

    # Resolves warning error with model_id:
    #     Field "model_id" has conflict with protected namespace "model_".
    #     You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    #     warnings.warn(
    class Config:
        protected_namespaces = ()

class texttosqlRequest(BaseModel):
    question: str = Field(title="NL Question", description="Question asked by the user.")
    dbtype: str = Field(title="Database Type", description="Database Type for Text To SQL")
    llm_params: Optional[LLMParams] = LLMParams()