# Watsonx.ai RAG Application

This application exposes APIs to help set up a RAG pipeline using **watsonx Discovery** (Elasticsearch) as the document respository.

This README will guide you through the steps to deploy the project locally, on OpenShift or IBM Code Engine. Additionally, you will learn how to access the Swagger documentation once the project is deployed.

## Deploying the application

### Deploying locally

To run application on your local machine, follow these steps:

1. Navigate to the project directory:

    ```bash
    cd RAG-LLM-Service/application
    ```

3. Create a Python Enviroment, Activate it, and Install Requirements:

    ```bash
    python -m venv assetEnv
    source assetEnv/bin/activate
    python -m pip install -r requirements.txt
    ```

4. Update your secrets:

    Copy `env` to `.env` and fill in the variables with your url, passwords, and apikeys.

    See the `env` file for more details on how to find the required values.

5. Start the project:

    ```bash
    python prereqs.py
    python app.py
    ```

6. URL access:

    The url, for purposes of using cURL is http://0.0.0.0:4050.

    To access Swagger go to http://0.0.0.0:4050/docs

### Deploying onto Code Engine

You can deploy this application onto [IBM Cloud Code Engine](https://cloud.ibm.com/docs/codeengine?topic=codeengine-getting-started) using Terraform scripts. See the steps [here.](./codeengine-setup/README.md)

### Deploying onto OpenShift

You can deploy this application onto a provisioned [Red Hat OpenShift](https://cloud.ibm.com/docs/openshift?topic=openshift-getting-started) cluster. See the steps [here.](./openshift-setup/README.md)

## Using the application APIs

After deploying the application, you can now test the API

### Test from Swagger

Open Swagger by going to `<url>/docs`.

#### ingestDocs

The `ingestDocs` API retrieves documents from a connected Cloud Object Storage bucket, chunks them and ingests them into the specified index in the connected Elasticsearch database. It will create the index and pipeline if they do not exist.

The specified model must be downloaded and deployed into the Elasticsearch instance.
1. Authenticate the `ingestDocs` api by clicking the lock button to the right.  Enter the value you added for the `RAG_APP_API_KEY`.

3. Click the `Try it out` button and customize your request body:
    ```
    {
      "GUID": "string",
      "title": "string",
      "URL": "string",
      "content": "string",
      "tags": [
        "string"
          ],
      "updated_date": "2024-09-26T19:33:28.221Z",
      "view_security_roles": [
        "string"
          ],
      "es_index_name": "string",
      "es_pipeline_name": "string",
      "chunk_size": "512",
      "chunk_overlap": "256",
      "es_model_name": ".elser_model_2_linux-x86_64",
      "es_model_text_field": "text_field",
      "es_index_text_field": "content"
    }
    ```

    **NOTE**: This api will append documents to an existing index and pipeline with the same name if it finds them. 

    If you want to delete the documents in an index before running this api:

    ```bash
    curl -X POST "<url:port>>/<index-name>/_delete_by_query" -H 'Content-Type: application/json' -d'{ "query":{ "match_all":{}}}' -k -u <userid>:<pwd>
    ```
    To delete an index:
    ```bash
    curl -X DELETE "<url:port>>/<index-name>?pretty" -H 'Content-Type: application/json' -k -u <userid>:<pwd>
    ```
    To delete a pipeline:
    ```bash
    curl -X DELETE "<url:port>/_ingest/pipeline/my-pipeline-id?pretty" -H 'Content-Type: application/json' -k -u <userid>:<pwd>
    ```
    
#### queryLLM
The `queryLLM` API queries a connected Elasticsearch database then sends the returned text into **watsonx.ai** using the designated LLM to return a natural language response.

1. Authenticate the `queryLLM` api by clicking the lock button to the right.  Enter the value you added for the `RAG_APP_API_KEY`.

3. Click the `Try it out` button and customize your request body:
    ```
    {
      "question": "string",
      "es_index_name": "string",
      "es_index_text_field": "body_content_field",
      "es_model_name": ".eelser_model_2_linux-x86_64",
      "es_model_text_field": "ml.tokens",
      "llm_instructions": "[INST]<<SYS>>You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Be brief in your answers. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\\'''t know the answer to a question, please do not share false information. <</SYS>>\nGenerate the next agent response by answering the question. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities and use the tiles of documents to separate between topics or domains. Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer.\n{context_str}<</SYS>>\n\n{query_str} Answer with no more than 150 words. If you cannot base your answer on the given document, please state that you do not have an answer. [/INST]",
      "num_results": "5",
      "llm_params": {
        "model_id": "meta-llama/llama-3-70b-instruct",
        "inputs": [],
        "parameters": {
          "decoding_method": "greedy",
          "max_new_tokens": 500,
          "min_new_tokens": 1,
          "moderations": {
            "hap_input": "true",
            "hap_output": "true",
            "threshold": 0.75
          },
          "repetition_penalty": 1.1,
          "temperature": 0.7,
          "top_k": 50,
          "top_p": 1
        }
      },
      "filters": {
        "date": "2022-01-01",
        "file_name": "test.pdf"
      }
    }
    ```

    At a minimum, specify:
    ```
    {
      "question": "<your question>",
      "es_index_name": "<your index>"
    }
    ```
    All other values have defaults, you can adjust the other parameters to improve your results.

    NOTE: The `filters` tag allows you to narrow down which documents to search on. You can specify from fields available within the document metadata. Remove this element if you don't want to filter on metadata.
   

### Test from cURL

To execute this api from command line, use this command: 
```
curl --location '<application url>/queryLLM' \
--header 'Content-Type: application/json' \
--header 'RAG-APP-API-Key: <your custom RAG-APP-API-KEY value>' \
--data '{
  "question": "string"
}'
```
### Test from Postman

1. Open a new tab and from the request type dropdown, select POST. In the url, paste your url (in this example, it's localhost): `http://127.0.0.1:4050/queryLLM`

2. Under Authorization, choose type **API Key**, add the following key/value: `RAG-APP-API-Key`/`<value for RAG_APP_API_KEY from .env>`

3. Under Body, select `raw` and paste the following json:
```
{
  "question": "<your question>",
}
```
4. Hit the blue `SEND` button and wait for your result.
