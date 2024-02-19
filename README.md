# A Dockerized Python Flask app on top of Openshift 4

## create project

oc new-project <project-name>

## Build and Push to Docker Hub from Openshift 4
```
$ oc new-build --strategy docker --binary --name=watsonx-discovery-rag

$ oc start-build watsonx-discovey-rag  --from-dir=. --follow --wait
```

## Deploy to OpenShift 4
```
$ oc new-app watsonx-discovery-rag  --name=watsonx-discovery-rag 
```

## Expose a Secure URL for this Flask app
```
$ oc create route edge --service=watsonx-discovery-rag
```
