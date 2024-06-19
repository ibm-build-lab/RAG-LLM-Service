import json
import os
import warnings

from dotenv import load_dotenv

# ElasticSearch
from elasticsearch import Elasticsearch, helpers, exceptions

load_dotenv()
warnings.filterwarnings("ignore")

# wxd creds
wxd_creds = {
    "username": os.environ.get("WXD_USERNAME"),
    "password": os.environ.get("WXD_PASSWORD"),
    "wxdurl": os.environ.get("WXD_URL")
}

# Create a global client connection to elastic search
es_client = Elasticsearch(
    wxd_creds["wxdurl"],
    basic_auth=(wxd_creds["username"], wxd_creds["password"]),
    verify_certs=False,
    request_timeout=3600,
)

def ingest_docs(filename_path):
    filename = os.path.splitext(filename_path)[0]

    es_index_name     = f'{filename}-index'
    es_pipeline_name  = f'{filename}-pipeline'
    es_model_name     = ".elser_model_2"
    es_model_text_field = "text_field"
    es_index_text_field = "body_content_field"

    with open(filename_path, 'r') as file:
        data = json.load(file)

    processedFAQ = []
    for item in data:
        combined_string = f"{item[0]}\n{item[1]}"
        processedFAQ.append(combined_string)

    # Pipeline must occur before index due to pipeline dependency
    create_inference_pipeline(es_client, es_pipeline_name, es_index_text_field, es_model_text_field, es_model_name)
    create_index(es_client, es_index_name, es_index_text_field, es_pipeline_name)

    for idx, doc in enumerate(data):
        actions = []
        for doc in data:
            action = {
                "_index": es_index_name,
                "_source": {
                    'body_content_field': str(doc),
                    'file_name': filename_path,
                    'metadata': {
                        "file_name": filename_path,
                        "_node_type": "TextNode"
                    }
                },
                "pipeline": es_pipeline_name
            }
            actions.append(action)

    try:
        response = helpers.bulk(es_client, actions)
        print(f"Bulk indexing completed with {response[0]} successes and {response[1]} errors.")
    except helpers.BulkIndexError as e:
        print("A bulk indexing error occurred:")
        for i, error_detail in enumerate(e.errors):
            print(f"Error {i+1}: {error_detail}")


def create_index(client, index_name, esIndexTextField, pipeline_name):
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
        if client.indices.exists(index=index_name):
            print("Deleting the existing index with same name")
            client.indices.delete(index=index_name)
        response = client.indices.create(index=index_name, body=index_config)
        print(response)
    except Exception as e:
        print(f"An error occurred when creating the index: {e}")
        response = e
        pass
    return response


def create_inference_pipeline(client, pipeline_name, esIndexTextField, esModelTextField, esModelName):
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
        if client.ingest.get_pipeline(id=pipeline_name):
            print("Deleting the existing pipeline with same name")
            client.ingest.delete_pipeline(id=pipeline_name)
    except:
        pass
    response = client.ingest.put_pipeline(id=pipeline_name, body=pipeline_config)
    return response

if __name__ == "__main__":
    filename = "temp.json"
    ingest_docs(filename)