import json
import os, getpass
import pandas as pd
import uvicorn
import sys
# import nest_asyncio

from utils import CloudObjectStorageReader, CustomWatsonX, create_sparse_vector_query_with_model, create_sparse_vector_query_with_model_and_filter
from dotenv import load_dotenv

# Fast API
from fastapi import FastAPI, Form, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# wx.ai
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning import APIClient

# ElasticSearch
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError

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

generate_params = {
    GenParams.MAX_NEW_TOKENS: 250,
    GenParams.DECODING_METHOD: "greedy",
    GenParams.STOP_SEQUENCES: ['END',';',';END'],
    GenParams.REPETITION_PENALTY: 1
}


@app.get("/")
def index():
    return {"Hello": "World"}

@app.post("/ingestDocs")
async def ingestDocs(request: ingestRequest)->ingestResponse:
    cosBucketName   = request.bucket_name
    chunkSize       = request.chunk_size
    chunkOverlap    = request.chunk_overlap
    esIndexName     = request.es_index_name
    esPipelineName  = request.es_pipeline_name
    esModelName     = request.es_model_name
    esModelTextField = request.es_model_text_field
    esIndexTextField = request.es_index_text_field
    # Metadata to add to nodes, could be anything from the user, maybe a list?
    metadataFields    = request.metadata_fields

    try: 
        cos_reader = CloudObjectStorageReader(
            bucket_name = cosBucketName,
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
        await create_inference_pipeline(async_es_client, esPipelineName, esIndexTextField, esModelTextField, esModelName)
        await create_index(async_es_client, esIndexName, esIndexTextField, esPipelineName)

        Settings.embed_model = None
        Settings.llm = None
        Settings.node_parser = SentenceSplitter.from_defaults(
            chunk_size=chunkSize, chunk_overlap=chunkOverlap
        )

        vector_store = ElasticsearchStore(
            es_client=async_es_client,
            index_name=esIndexName,
            text_field=esIndexTextField
        )

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            show_progress=True,
        )

        return ingestResponse(response="success: number of documents loaded " + len(documents))
    except Exception as e:
        return ingestResponse(response = json.dumps({"error": repr(e)}))


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

# This function is NOT using the WML library to call the LLM. It is using
# llama_index
@app.post("/queryLLM")
def queryLLM(request: queryLLMRequest)->queryLLMResponse:
    question         = request.question
    index_name       = request.es_index_name
    index_text_field = request.es_index_text_field
    es_model_name    = request.es_model_name
    num_results      = request.num_results
    llm_params       = request.llm_params
    document         = request.document

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
    Settings.llm = CustomWatsonX(
        credentials=wml_credentials,
        project_id=project_id,
        model_id=llm_params.model_id,
        validate_model_id=False,
        additional_kwargs=llm_params.parameters.dict(),
    )
    Settings.embed_model = None

    # Create a client connection to elastic search
    async_es_client = AsyncElasticsearch(
        wxd_creds["wxdurl"],
        basic_auth=(wxd_creds["username"], wxd_creds["password"]),
        verify_certs=False,
        request_timeout=3600,
    )
    
    # Create a vector store using the elastic client
    vector_store = ElasticsearchStore(
        es_client=async_es_client,
        index_name=index_name,
        text_field=index_text_field
    )

    # Retrieve an index of the ingested documents in the vector store
    # for later retrieval and querying
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Create a retriever object using the index and setting params
    
    if document: 
        filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                    key="file_name", operator=FilterOperator.EQ, value=document
                ),
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

    # Format the data
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


@app.post("/getDocs")
def getDocs():
    return {"Not":"Implemented yet"}

@app.post("/watsonx")
def watsonx(input, promptType, model):
    
    #GRANITE_13B_CHAT = 'ibm/granite-13b-chat-v1'
    model = Model(
    model_id=model,
    params=generate_params,
    credentials={
        "apikey": os.environ.get("IBM_CLOUD_API_KEY"),
        "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id=os.environ.get("WX_PROJECT_ID")
    )

    request_data = request.get_json()

    key = os.environ.get("IBM_CLOUD_API_KEY")

    promptText=open(promptType,"r")

    prompt=promptText.read()

    finalInput=prompt + "Input: " + input

    generated_responsBaseModele = model.generate(prompt=finalInput)
    response=generated_response['results'][0]['generated_text']
 
    #return render_template('index.html', message="SQL " + response)
    #sql = [{'SQL': response}]
    #print(sql)

    #output_json_str = queryexec(response.replace('\n\n', '').replace(';',''))

    return response

@app.post("/setup_index")
def setupIndex():
    return {"Not":"Implemented yet"}


if __name__ == '__main__':
    if 'uvicorn' not in sys.argv[0]:
        uvicorn.run("app:app", host='0.0.0.0', port=3000, reload=True)