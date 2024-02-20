from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# ElasticSearch
from elasticsearch import AsyncElasticsearch
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    get_response_synthesizer,
    PromptTemplate,
)
from llama_index.llms import WatsonX
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

import prestodb
import json
import os, getpass
import pandas as pd
import jaydebeapi

import utils

# from elasticsearch.exceptions import NotFoundError
# from llama_index import download_loader
# from llama_index.readers.base import BaseReader
# from llama_index.schema import BaseNode, Document, MetadataMode, TextNode
# from llama_index.vector_stores.elasticsearch import (
#     ElasticsearchStore,
#     _to_elasticsearch_filter,
#     _to_llama_similarities,
# )
# from llama_index.vector_stores.types import (
#     VectorStoreQuery,
#     VectorStoreQueryMode,
#     VectorStoreQueryResult,
# )
# from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict



app = Flask(__name__)

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

@app.route("/")
def index():
    return render_template('index.html', message="Hello PRS..!!")

@app.route("/ingestDocs", methods=['POST'])
def ingestDocs():
    return '{"key":"Hello World"}'


# This function is NOT using the WML library to call the LLM. It is using
# llama_index
@app.route("/getDocsWithLLM", methods=['POST'])
def getDocsWithLLM():
    
    question = "What are some key features of Watsonx.governance?"
    
    # Format payload for later query
    payload = {
        "input_data": [
            {"fields": ["Text"], "values": [[question]]}
        ]
    }

    # Attempt to connect to ElasticSearch and call Watsonx for a response
    try:
        # Not really sure why it's like this.. need to further test..
        user_query = payload["input_data"][0]["values"][0][0]

        # from_parameter_set = external_asses["paramset_values"] # No idea what this is
        index_name = "es_index_name"
        index_text_field = "body_content_field"
        es_model_name = "elser_model_1"
        max_overlap_score = ""
        concatenated_overlap_score = ""
        num_results = "es_query_num_results"

        llm_params = {
            "model_id": "meta-llama/llama-2-70b-chat",
            "inputs": [],
            "parameters": {
                "decoding_method": "greedy",
                "min_new_tokens": 1,
                "max_new_tokens": 500,
                "moderations": {
                    "hap": {
                        "input": true,
                        "threshold": 0.75,
                        "output": true
                    }
                }
            }
        }

        llm_instructions = "[INST]<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'\''t know the answer to a question, please do not share false information. <</SYS>>\nGenerate the next agent response by answering the question. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities and use the tiles of documents to separate between topics or domains. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer.\n{context_str}<</SYS>>\n\n{query_str} Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer. [/INST]"

        prompt_template = PromptTemplate(llm_instructions)

        async_es_client = AsyncElasticsearch(
            wxd_creds["url"],
            basic_auth=(wxd_creds["username"], wxd_creds["password"]),
            verify_certs=False,
            request_timeout=3600,
        )
        vector_store = utils.ElserElasticsearchStore(
            es_client=async_es_client,
            index_name=index_name,
            pipeline_name="_",
            model_id=model_name,
            text_field=index_text_field,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            service_context=ServiceContext.from_defaults(embed_model=None, llm=None),
        )
        retriever = VectorIndexRetriever(
            index=index,
            vector_store_query_mode="sparse",
            similarity_top_k=num_results,
        )

        llm = WatsonX(
            credentials=wml_credentials,
            space_id=project_id,
            model_id=llm_params["model_id"],
            max_new_tokens=llm_params["parameters"]["max_new_tokens"],
            additional_kwargs=llm_params["parameters"],
        )
        response_synthesizer = get_response_synthesizer(
            service_context=ServiceContext.from_defaults(embed_model=None, llm=llm),
            text_qa_template=prompt_template,
        )

        query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer
        )
    
   
        response = query_engine.query(user_query)
        (
            llm_response,
            max_overlap_score,
            concatenated_overlap_score,
        ) = utils.detect_and_process_hallucination(
            response.response, response.source_nodes
        )
        scoring_response = {
            "predictions": [{"llm_response": llm_response}],
            "llm_response": llm_response,
            "references": [node.to_dict() for node in response.source_nodes],
            "max_overlap_score": round(max_overlap_score, 2),
            "concatenated_overlap_score": round(concatenated_overlap_score, 2),
        }
        return scoring_response

    except Exception as e:
        return {"predictions": [{"error": repr(e)}]}

    return '{"key":"issue in getDocsWithLLMs"}'

@app.route("/getDocs", methods=['POST'])
def getDocs(answer):
    imageURL=""

    if answer.find("187800") >1:
      imageURL="https://yasserssandbox-donotdelete-pr-01xjiqorpvqifx.s3.us-south.cloud-object-storage.appdomain.cloud/q1.png"
    elif answer.find("72800") >1:
      imageURL="https://yasserssandbox-donotdelete-pr-01xjiqorpvqifx.s3.us-south.cloud-object-storage.appdomain.cloud/q2.png"
    elif answer.find("BMW") >1:
      imageURL="https://yasserssandbox-donotdelete-pr-01xjiqorpvqifx.s3.us-south.cloud-object-storage.appdomain.cloud/q4.png"
    elif answer.find("30000") >1:
      imageURL="https://yasserssandbox-donotdelete-pr-01xjiqorpvqifx.s3.us-south.cloud-object-storage.appdomain.cloud/q3.png"
    else:
      imageURL="https://yasserssandbox-donotdelete-pr-01xjiqorpvqifx.s3.us-south.cloud-object-storage.appdomain.cloud/q1.png"
    return imageURL

@app.route("/watsonx", methods=['POST'])
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

@app.route("/setup_index", methods=['POST'])
def setupIndex():
     
    request_data = request.get_json()
    query = request_data['Query']

    conn = jaydebeapi.connect("com.ibm.db2.jcc.DB2Driver", "jdbc:db2://b869522f-19c9-4c7c-9b2a-735b59a54ead.c1ogj3sd0tgtu0lqde00.databases.appdomain.cloud:32002/bludb:user=30734ea0;password=xG2dNqaTiTazCgQC;sslConnection=true;",None, "db2jcc4.jar")
  
    '''
    conn = prestodb.dbapi.connect(
       host='ibm-lh-lakehouse-presto-01-presto-svc-cpd-instance.apps.65326fcf94ee63001721417c.cloud.techzone.ibm.com',
       port=443,
       user='admin',
       #catalog='tpch',
       #schema='tiny',
       catalog='ben',
       schema='ben',
       http_scheme='https',
       auth=prestodb.auth.BasicAuthentication('admin', '1BtJuGhTx4AT')
    )
    '''
    #conn._http_session.verify = "tls.crt"
    cur = conn.cursor()
    
    #cur.execute("SELECT * FROM prsgroup.rzy62361.country")
    cur.execute(query)
    rows = cur.fetchall()

    queryResults = pd.DataFrame.from_records(rows, columns = [i[0] for i in cur.description])
    
    queryResults2 = queryResults.to_json(orient = 'columns')
    
    queryResults2= json.loads(queryResults2)
    print(queryResults2)
    output_json = {}

    keys = list(queryResults2.keys())
     
    result = []

    for i in range(len(queryResults2['Year'])):
        obj = {}
        for key in keys:
            obj[key] = queryResults2[key][str(i)]
        result.append(obj)

    #for key in keys:
    #    output_json[key] = queryResults2[key]["0"]

    output_json_str = json.dumps(result)
    print(output_json_str)
  
    return render_template('index.html', message=output_json_str)


def helpFunction():
    return ""

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
