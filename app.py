import json
import os, getpass
import pandas as pd
import uvicorn
import sys
import utils

from utils import ElserElasticsearchStore, CloudObjectStorageReader
from dotenv import load_dotenv
# IBM COS
import ibm_boto3
from ibm_botocore.client import Config, ClientError

# Fast API
from fastapi import FastAPI, Form, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# wx.ai
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# ElasticSearch
from elasticsearch import AsyncElasticsearch
from llama_index import ServiceContext, VectorStoreIndex, get_response_synthesizer, PromptTemplate

# Vector Store / WatsonX connection
from llama_index.llms import WatsonX
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import SentenceSplitter


# Custom type classes
# from customTypes.classificationRequest import classificationRequest
# from customTypes.summarizationRequest import summarizationRequest
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

generate_params = {
    GenParams.MAX_NEW_TOKENS: 250,
    GenParams.DECODING_METHOD: "greedy",
    GenParams.STOP_SEQUENCES: ['END',';',';END'],
    GenParams.REPETITION_PENALTY: 1
}


@app.get("/")
def index():
    return {"Hello": "World1"}

@app.post("/ingestDocs")
async def ingestDocs():
    # Create resource
    cos = ibm_boto3.resource("s3",
        ibm_api_key_id=os.environ.get("COS_IBM_CLOUD_API_KEY"),
        ibm_service_instance_id=os.environ.get("COS_INSTANCE_ID"),
        ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
        config=Config(signature_version="oauth"),
        endpoint_url="https://s3.us-south.cloud-object-storage.appdomain.cloud"
    )
    ibm_api_key_id=os.environ.get("COS_IBM_CLOUD_API_KEY")
    print("IBM API KEY: " +str(ibm_api_key_id))

    files = cos.Bucket("celonis").objects.all()
    print(files)
    cos_reader = utils.CloudObjectStorageReader(
        bucket_name = os.environ.get("BUCKET_NAME"),
        credentials = {
            "apikey": os.environ.get("COS_IBM_CLOUD_API_KEY"),
            "service_instance_id": os.environ.get("COS_INSTANCE_ID")
        },
        hostname = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
    )

    print(cos_reader.list_files())
 
    documents = await cos_reader.load_data()
    print(f"Total documents: {len(documents)}\nExample document:\n{documents[0]}")
    ingestion_pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter.from_defaults(
                chunk_size=512, chunk_overlap=256
            ),
        ]
    )
    nodes = ingestion_pipeline.run(documents=documents)
    nodes[0]

    for node in nodes:
        node.metadata["url"] = "https://ibm.box.com"
    print(f"Total Nodes: {len(nodes)}\nExample node:\n{nodes[0]}")



    async_es_client = AsyncElasticsearch(
            wxd_creds["wxdurl"],
            basic_auth=(wxd_creds["username"], wxd_creds["password"]),
            verify_certs=False,
            request_timeout=3600,
    )

    await async_es_client.info()
    
    index_config = {
        "mappings": {
            "properties": {"ml.tokens": {"type": "rank_features"}, "body_content_field": {"type": "text"}}
        }
    }
    
    print( index_config)
    
    pipeline_config = {
        "description": "Inference pipeline using elser model",
        "processors": [
            {
                "inference": {
                    "field_map": {"body_content_field": "text_field"},
                    "model_id": ".elser_model_1",
                    "target_field": "ml",
                    "inference_config": {"text_expansion": {"results_field": "tokens"}},
                }
            },
        {
            "set": {
                "field": "file_name",
                "value": "{{metadata.filename}}"
            }
        },
        {
            "set": {
                "field": "url",
                "value": "{{metadata.url}}"
            }
        },
        {
            "append": {
                "field": "_source._ingest.processors",
                "value": [
                        {
                            "model_version": "10.0.0",
                            "pipeline": "pipeline-created-in-watson-studio-notebook",
                            "processed_timestamp": "{{{ _ingest.timestamp }}}",
                            "types": ["pytorch", "text_expansion"],
                        }
                    ],
                }
            },
        ],
        "version": 1,
    }
    await create_index(async_es_client, "index-created-in-watson-studio-notebook", index_config)
    await create_inference_pipeline(async_es_client, "pipeline-created-in-watson-studio-notebook", pipeline_config)
    
    vector_store = ElserElasticsearchStore(
         es_client=async_es_client,
         index_name=os.environ.get("INDEX_NAME"),
         pipeline_name=os.environ.get("PIPELINE_NAME"),
         model_id=os.environ.get("EMBEDDING_MODEL_NAME"),
         text_field=os.environ.get("INDEX_TEXT_FIELD"),
         batch_size=10
    )
    added_node_ids = await vector_store.async_add(nodes)

    print("added node ids: " + str(added_node_ids))
    #print(f"Added {len(added_node_ids)} nodes to index index-created-in-watson-studio-notebook using pipeline pipeline-created-in-watson-studio-notebook")

    return {"success":"true"}


async def create_index(client, index_name, index_settings):
    print("Creating the index...")
    try:
        if await client.indices.exists(index=index_name):
            print("Deleting the existing index with same name")
            await client.indices.delete(index=index_name)
        response = await client.indices.create(index=index_name, body=index_settings)
        print(response)
    except Exception as e:
        print(f"An error occurred when creating the index: {e}")
        response = e
        pass
    return response


async def create_inference_pipeline(client, pipeline_name, pipeline_settings):
    print("Creating the inference pipeline...")
    try:
        if await client.ingest.get_pipeline(id=pipeline_name):
            print("Deleting the existing pipeline with same name")
            await client.ingest.delete_pipeline(id=pipeline_name)
    except:
        pass
    response = await client.ingest.put_pipeline(id=pipeline_name, body=pipeline_settings)
    print(response)
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

    # Sets the llm instruction if the user provides it
    if not request.llm_instructions:
        llm_instructions = os.environ.get("LLM_INSTRUCTIONS")
    else:
        llm_instructions = request.llm_instructions

    # question = "What are some key features of Watsonx.governance?"
    # index_name = "index-created-in-watson-studio-notebook"

    # Format payload for later query
    payload = {
        "input_data": [
            {"fields": ["Text"], "values": [[question]]}
        ]
    }

    # Attempt to connect to ElasticSearch and call Watsonx for a response
    try:
        # Setting up the structure of the payload for the query engine
        user_query = payload["input_data"][0]["values"][0][0]

        # Create the prompt template based on llm_instructions
        prompt_template = PromptTemplate(llm_instructions)

        # Create a client connection to elastic search
        async_es_client = AsyncElasticsearch(
            wxd_creds["wxdurl"],
            basic_auth=(wxd_creds["username"], wxd_creds["password"]),
            verify_certs=False,
            request_timeout=3600,
        )
        
        # Create a vector store using the elastic client
        vector_store = utils.ElserElasticsearchStore(
            es_client=async_es_client,
            index_name=index_name,
            pipeline_name="_",
            model_id=es_model_name,
            text_field=index_text_field,
        )

        # Retrieve an index of the ingested documents in the vector store
        # for later retrieval and querying
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=ServiceContext.from_defaults(embed_model=None, llm=None),
        )

        # Create a retriever object using the index and setting params
        retriever = VectorIndexRetriever(
            index=index,
            vector_store_query_mode="sparse",
            similarity_top_k=num_results,
        )

        # Create the watsonx LLM object that will be used for the RAG pattern
        llm = WatsonX(
            credentials=wml_credentials,
            project_id=project_id,
            model_id=llm_params.model_id,
            max_new_tokens=llm_params.parameters.max_new_tokens,
            additional_kwargs=llm_params.parameters.dict(),
        )

        #
        response_synthesizer = get_response_synthesizer(
            service_context=ServiceContext.from_defaults(embed_model=None, llm=llm),
            text_qa_template=prompt_template,
        )

        # Create the query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer
        )
        
        # Finally query the engine with the user question
        response = query_engine.query(user_query)

        # Format the data
        data_response = {
            "llm_response": response.response,
            "references": [node.to_dict() for node in response.source_nodes]
        }

        return queryLLMResponse(**data_response)

    except Exception as e:
        return queryLLMResponse(
            llm_response = "",
            references=[{"error": repr(e)}]
        )


@app.post("/getDocs")
def getDocs():
    
    return ""

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

    generated_response = model.generate(prompt=finalInput)
    response=generated_response['results'][0]['generated_text']
 
    #return render_template('index.html', message="SQL " + response)
    #sql = [{'SQL': response}]
    #print(sql)

    #output_json_str = queryexec(response.replace('\n\n', '').replace(';',''))

    return response

@app.post("/setup_index")
def setupIndex():
    return ""

def helpFunction():
    print("hi")
    return ""


if __name__ == '__main__':
    if 'uvicorn' not in sys.argv[0]:
        uvicorn.run("app:app", host='0.0.0.0', port=3000, reload=True)