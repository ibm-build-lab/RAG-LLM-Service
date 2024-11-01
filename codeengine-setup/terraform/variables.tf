
variable "region" {
  description = "Region where Code Engine project will be created"
  type        = string
  default     = "us-south"
}

variable "resource_group" {
  type    = string
  default = "Default"
  description = "Resource group where Code Engine project and application will reside. Must already exist"
}

variable "source_url" {
  type    = string
  default = "https://github.com/ibm-build-lab/RAG-LLM-Service.git"
  description = "Git repo source name"
}

variable "source_revision" {
  type    = string
  default = "main"
  description = "Git repo branch name"
}

variable "source_context_dir" {
  type    = string
  default = "application"
  description = "Subdirectory where Dockerfile and application files are located"
}

variable "cr_namespace" {
  type    = string
  default = ""
  description = "Container Registry namespace. Must be between 4 and 24 characters"
}

variable "cr_imagename" {
  type        = string
  description = "Build image"
  default = "rag-llm"
}

variable "ce_project_name" {
  type    = string
  default = ""
  description = "Name of Code Engine project"
}

variable "ce_buildsecret" {
  type        = string
  description = "Code Engine build secret"
  default = "buildsecret"
}

variable "ce_buildname" {
  type        = string
  description = "Code Engine build name"
  default = "rag-llm-build"
}

variable "ce_appname" {
  type        = string
  description = "Code Engine application name"
  default = "rag-llm-service"
}

variable "cos_ibm_cloud_api_key" {
  type        = string
  description = "COS Bucket API Key. Service Credentials -> <bucket name>"
  default = ""
}

variable "cos_instance_id" {
  type        = string
  description = "COS Instance ID. Service Credentials -> <bucket name>. Ends with ::"
  default = ""
}

variable "cos_endpoint_url" {
  type        = string
  description = "COS endpoint. Open bucket, go into Configuration tab, public endpoint, make sure to add the https://<URL>"
  default = "https://s3.us-east.cloud-object-storage.appdomain.cloud"
}

variable "rag_app_api_key" {
  type        = string
  description = "RAG APP User Created Key, used for API authentication"
  default = ""
}

variable "wx_project_id" {
  type        = string
  description = "watsonx.ai project id. Open the project, go to Management->General->Details"
  default = ""
}

variable "wx_url" {
  type        = string
  description = "watsonx URL"
  default = "https://us-south.ml.cloud.ibm.com"
}

variable "wxd_username" {
  type        = string
  description = "Elasticsearch user"
  default = ""
}

variable "wxd_password" {
  type        = string
  description = "Elasticsearch password"
  default = ""
}

variable "wxd_url" {
  type        = string
  description = "Elasticsearch URL and port. Get from Elasticsearch DB Service->Overview->Endpoints->HTTPS"
  default = ""
}

variable "wd_api_key" {
  type        = string
  description = "Watson Discovery apikey. Get from WD Service Credentials"
  default = "************"
}

variable "wd_url" {
  type        = string
  description = "Watson Discovery URL. Get from WD Service Credentials"
  default = "https://<url>"
}


