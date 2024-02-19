from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
import prestodb
import json
import os, getpass
import pandas as pd
import jaydebeapi

app = Flask(__name__)

load_dotenv()

#Token to IBM Cloud
ibm_cloud_api_key = os.environ.get("IBM_CLOUD_API_KEY")
project_id = os.environ.get("WX_PROJECT_ID")

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

'''
@app.route("/nl", methods=['POST'])
def nlquery():
    
    request_data = request.get_json()
    query = request_data['NL']
    history = request_data['History']

    print(request)
    key = os.environ.get("IBM_CLOUD_API_KEY")

    promptText=open("prompt.txt","r")
    prompt=promptText.read()
    q=prompt + "\n" +  query
    generated_response = model.generate(prompt=q)
    response=generated_response['results'][0]['generated_text']
 
    #return render_template('index.html', message="SQL " + response)
    sql = [{'SQL': response}]
    print(sql)

    output_json_str = queryexec(response.replace('\n\n', '').replace(';',''))

    return output_json_str
'''
@app.route("/nl", methods=['POST'])
def nlquery():
    
    request_data = request.get_json()
    query = request_data['NL']
    history = request_data['history']

    if history == "":
        #Call watosnx.ai to generate SQL
        watsonxSQLResponse = watsonx (query,"promptSQL", "meta-llama/llama-2-13b-chat")
    else:
        watsonxSQLResponse = watsonx (history + " "+ query,"promptSQL", "meta-llama/llama-2-13b-chat")

    #return render_template('index.html', message="SQL " + response)
    sql = [{'SQL': watsonxSQLResponse}]
    print(sql)
    #execute the SQL statement
   
    queryfromwatsonx = watsonxSQLResponse.replace('\nOutput: ','').replace(';','')
    output_json_str = queryexec(queryfromwatsonx)

    
    print(output_json_str.get("answer"))
    
    #Call watosnx.ai to generate NL Response
    watsonxJSONResponse = watsonx (output_json_str.get("answer"),"promptJSON","meta-llama/llama-2-13b-chat")
    print(watsonxJSONResponse)
    output_json_str["nl"]=watsonxJSONResponse.replace('\nOutput:','').replace('END','')

    image = getImage(output_json_str.get("answer"))
    output_json_str["history"]=history
    output_json_str["image"]=image
    return output_json_str

@app.route("/getImage", methods=['POST'])
def getImage(answer):
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

@app.route("/query", methods=['POST'])
def query():
     
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

@app.route("/queryexec", methods=['POST'])
def queryexec(query):
   
    print("exec query:" + query)

   # conn = jaydebeapi.connect("com.ibm.db2.jcc.DB2Driver", "jdbc:db2://b869522f-19c9-4c7c-9b2a-735b59a54ead.c1ogj3sd0tgtu0lqde00.databases.appdomain.cloud:32002/bludb:currentSchema=BEN;user=30734ea0;password=xG2dNqaTiTazCgQC;sslConnection=true;",None, "db2jcc4.jar")
    conn = jaydebeapi.connect("com.ibm.db2.jcc.DB2Driver", "jdbc:db2://98266017-7adf-445a-90ed-cf8e64c0f41e.bv7c3o6d0vfhru3npds0.databases.appdomain.cloud:31968/bludb:currentSchema=BEN;user=db5318c8;password=mVr8LOYkcABt6Zk4;sslConnection=true;",None, "db2jcc4.jar")
   
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
    #conn._http_session.verify = "tls.crt"
    '''
    cur = conn.cursor()
    
    #cur.execute("SELECT * FROM prsgroup.rzy62361.country")
    cur.execute(query)
    rows = cur.fetchall()
    op=""

    for row in rows:
        br="" 
        for i,col in enumerate(row):
            key=cur.description[i][0]
            br += "{}:{},".format(key,col)
        br = br[:-1]
        op += "{" + br + "}"

      #  op += watsonx.ai("{" + br + "}") + "\n"
    nl=""
    history=""
    image=""
    return dict(answer=op,query=query,nl=nl,history=history,image=image)


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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
