import json
import os
import uvicorn
import sys
import time

from utils import CloudObjectStorageReader, CustomWatsonX, create_sparse_vector_query_with_model, create_sparse_vector_query_with_model_and_filter
from dotenv import load_dotenv

# Fast API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ElasticSearch
from elasticsearch import Elasticsearch, AsyncElasticsearch

# Vector Store / WatsonX connection
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter, FilterOperator, MetadataFilter


# Custom type classes
from customTypes.ingestRequest import ingestRequest
from customTypes.ingestResponse import ingestResponse
from customTypes.queryLLMRequest import queryLLMRequest
from customTypes.queryLLMResponse import queryLLMResponse


app = FastAPI()

# Set up CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

#Token to IBM Cloud
ibm_cloud_api_key = os.environ.get("IBM_CLOUD_API_KEY")
project_id = os.environ.get("WX_PROJECT_ID")

# wxd creds
wxd_creds = {
    "username": os.environ.get("WXD_USERNAME"),
    "password": os.environ.get("WXD_PASSWORD"),
    "wxdurl": os.environ.get("WXD_URL")
}

# WML Creds
wml_credentials = {
    "url": os.environ.get("WX_URL"),
    "apikey": os.environ.get("IBM_CLOUD_API_KEY")
}

# COS Creds
cos_creds = {
    "cosIBMApiKeyId": os.environ.get("COS_IBM_CLOUD_API_KEY"),
    "cosServiceInstanceId": os.environ.get("COS_INSTANCE_ID"),
    "cosEndpointURL": os.environ.get("COS_ENDPOINT_URL")
}

# Create a global client connection to elastic search
async_es_client = AsyncElasticsearch(
    wxd_creds["wxdurl"],
    basic_auth=(wxd_creds["username"], wxd_creds["password"]),
    verify_certs=False,
    request_timeout=3600,
)

# Create a watsonx client cache for faster calls.
custom_watsonx_cache = {}


@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/ingestDocs")
async def ingestDocs(request: ingestRequest)->ingestResponse:
    cos_bucket_name   = request.bucket_name
    chunk_size        = request.chunk_size
    chunk_overlap     = request.chunk_overlap
    es_index_name     = request.es_index_name
    es_pipeline_name  = request.es_pipeline_name
    es_model_name     = request.es_model_name
    es_model_text_field = request.es_model_text_field
    es_index_text_field = request.es_index_text_field
    # TODO: Metadata to add to nodes, could be anything from the user, maybe a list?
    metadata_fields     = request.metadata_fields

    # try: 
    cos_reader = CloudObjectStorageReader(
        bucket_name = cos_bucket_name,
        credentials = {
            "apikey": cos_creds["cosIBMApiKeyId"],
            "service_instance_id": cos_creds["cosServiceInstanceId"]
        },
        hostname = cos_creds["cosEndpointURL"]
    )

    print(cos_reader.list_files())

    documents = await cos_reader.load_data()
    print(f"Total documents: {len(documents)}\nExample document:\n{documents[0]}")

    async_es_client = AsyncElasticsearch(
        wxd_creds["wxdurl"],
        basic_auth=(wxd_creds["username"], wxd_creds["password"]),
        verify_certs=False,
        request_timeout=3600,
    )

    await async_es_client.info()

    # Pipeline must occur before index due to pipeline dependency
    await create_inference_pipeline(async_es_client, es_pipeline_name, es_index_text_field, es_model_text_field, es_model_name)
    await create_index(async_es_client, es_index_name, es_index_text_field, es_pipeline_name)

    Settings.embed_model = None
    Settings.llm = None
    Settings.node_parser = SentenceSplitter.from_defaults(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    vector_store = ElasticsearchStore(
        es_client=async_es_client,
        index_name=es_index_name,
        text_field=es_index_text_field
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        show_progress=True,
        use_async=True
    )

    return ingestResponse(response="success: number of documents loaded " + str(len(documents)))
    # except Exception as e:
    #     return ingestResponse(response = json.dumps({"error": repr(e)}))


async def create_index(client, index_name, esIndexTextField, pipeline_name):
    print("Creating the index...")
    index_config = {
        "mappings": {
            "properties": {
                "ml.tokens": {"type": "rank_features"}, 
                esIndexTextField: {"type": "text"}}
        },
        "settings": {
            "index.default_pipeline": pipeline_name,
        }
    }
    try:
        if await client.indices.exists(index=index_name):
            print("Deleting the existing index with same name")
            await client.indices.delete(index=index_name)
        response = await client.indices.create(index=index_name, body=index_config)
        print(response)
    except Exception as e:
        print(f"An error occurred when creating the index: {e}")
        response = e
        pass
    return response


async def create_inference_pipeline(client, pipeline_name, esIndexTextField, esModelTextField, esModelName):
    print("Creating the inference pipeline...")
    pipeline_config = {
        "description": "Inference pipeline using elser model",
        "processors": [
            {
                "inference": {
                    "field_map": {esIndexTextField: esModelTextField},
                    "model_id": esModelName,
                    "target_field": "ml",
                    "inference_config": {"text_expansion": {"results_field": "tokens"}},
                }
            },
            {"set": {"field": "file_name", "value": "{{metadata.file_name}}"}},
            {"set": {"field": "url", "value": "{{metadata.url}}"}},
        ],
        "version": 1,
    }

    try:
        if await client.ingest.get_pipeline(id=pipeline_name):
            print("Deleting the existing pipeline with same name")
            await client.ingest.delete_pipeline(id=pipeline_name)
    except:
        pass
    response = await client.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)
    return response

# Uses Llama-index to obtain the context from an ES query
# which uses WML library underneath the hood via
# a CustomWatsonX class in utils.py
@app.post("/queryLLM")
def queryLLM(request: queryLLMRequest)->queryLLMResponse:

    question         = request.question
    index_name       = request.es_index_name
    index_text_field = request.es_index_text_field
    es_model_name    = request.es_model_name
    num_results      = request.num_results
    llm_params       = request.llm_params
    es_filters       = request.filters

    # Sets the llm instruction if the user provides it
    if not request.llm_instructions:
        llm_instructions = os.environ.get("LLM_INSTRUCTIONS")
    else:
        llm_instructions = request.llm_instructions

    # Format payload for later query
    payload = {
        "input_data": [
            {"fields": ["Text"], "values": [[question]]}
        ]
    }

    # Attempt to connect to ElasticSearch and call Watsonx for a response
    # try:
    # Setting up the structure of the payload for the query engine
    user_query = payload["input_data"][0]["values"][0][0]

    # Create the prompt template based on llm_instructions
    prompt_template = PromptTemplate(llm_instructions)

    # Create the watsonx LLM object that will be used for the RAG pattern
    Settings.llm = get_custom_watsonx(llm_params.model_id, llm_params.parameters.dict())
    Settings.embed_model = None

    # Create a vector store using the elastic client
    vector_store = ElasticsearchStore(
        es_client=async_es_client,
        index_name=index_name,
        text_field=index_text_field
    )

    # Retrieve an index of the ingested documents in the vector store
    # for later retrieval and querying
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    if es_filters: 
        print(es_filters)
        for k, v in es_filters.items():
            print(k)
            print(v)
        filters = MetadataFilters(
                filters=[
                    MetadataFilter(key=k,operator=FilterOperator.EQ, value=v) for k, v in es_filters.items()
            ]
        )
        
        query_engine = index.as_query_engine(
            text_qa_template=prompt_template,
            similarity_top_k=num_results,
            vector_store_query_mode="sparse",
            vector_store_kwargs={
                "custom_query": create_sparse_vector_query_with_model_and_filter(es_model_name, filters=filters)
            },
        )
    else:
        query_engine = index.as_query_engine(
            text_qa_template=prompt_template,
            similarity_top_k=num_results,
            vector_store_query_mode="sparse",
            vector_store_kwargs={
                "custom_query": create_sparse_vector_query_with_model(es_model_name)
            },
        )

    # Finally query the engine with the user question
    response = query_engine.query(user_query)

    data_response = {
        "llm_response": response.response,
        "references": [node.to_dict() for node in response.source_nodes]
    }

    return queryLLMResponse(**data_response)

    # except Exception as e:
    #     return queryLLMResponse(
    #         llm_response = "",
    #         references=[{"error": repr(e)}]
    #     )

def get_custom_watsonx(model_id, additional_kwargs):
    # Serialize additional_kwargs to a JSON string, with sorted keys
    additional_kwargs_str = json.dumps(additional_kwargs, sort_keys=True)
    # Generate a hash of the serialized string
    additional_kwargs_hash = hash(additional_kwargs_str)
    
    cache_key = f"{model_id}_{additional_kwargs_hash}"

    # Check if the object already exists in the cache
    if cache_key in custom_watsonx_cache:
        return custom_watsonx_cache[cache_key]

    # If not in the cache, create a new CustomWatsonX object and store it
    custom_watsonx = CustomWatsonX(
        credentials=wml_credentials,
        project_id=project_id,
        model_id=model_id,
        validate_model_id=False,
        additional_kwargs=additional_kwargs,
    )
    custom_watsonx_cache[cache_key] = custom_watsonx
    return custom_watsonx

if __name__ == '__main__':
    if 'uvicorn' not in sys.argv[0]:
        uvicorn.run("app:app", host='0.0.0.0', port=4050, reload=True)