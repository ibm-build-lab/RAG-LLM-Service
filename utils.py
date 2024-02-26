from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    AsyncGenerator,
)
from pathlib import Path

from llama_index.vector_stores.elasticsearch import (
    ElasticsearchStore,
    _to_elasticsearch_filter,
    _to_llama_similarities,
)
from llama_index import download_loader
from llama_index.schema import BaseNode, Document, MetadataMode, TextNode
from llama_index.readers.base import BaseReader

from llama_index.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

import requests
import re
import asyncio
import aiohttp
import logging
import tempfile


# def detect_and_process_hallucination(
#     llm_response,
#     reference_nodes,
#     max_overlap_threshold=from_parameter_set[
#         "hallucination_threshold_max_text_overlap"
#     ],
#     concatenated_overlap_threshold=from_parameter_set[
#         "hallucination_threshold_concatenated_text_overlap"
#     ],
# ):
#     context = [
#         {
#             "document_title": node.node.metadata["filename"],
#             "document_text": node.node.text,
#         }
#         for node in reference_nodes
#     ]
#     max_overlap_score, concatenated_overlap_score = utils.get_overlap_scores(
#         llm_response, context
#     )
#     is_hallucination = (
#         max_overlap_score < max_overlap_threshold
#         or concatenated_overlap_score < concatenated_overlap_threshold
#     )
#     if is_hallucination:
#         hallucination_prefix = "WARNING: Given the low max/concatenated text overlap score, LLM response may contain hallucinations. "
#         llm_response = f"{hallucination_prefix}{llm_response}"

#     return llm_response, max_overlap_score, concatenated_overlap_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_READER_NAMES = {
    ".pdf": "PDFReader",
    ".docx": "DocxReader",
    ".pptx": "PptxReader",
    ".txt": "UnstructuredReader",
    ".html": "UnstructuredReader",
}

class CloudObjectStorageReader(BaseReader):
    """
    A class used to interact with IBM Cloud Object Storage.

    This class inherits from the BasePydanticReader base class and overrides its methods to work with IBM Cloud Object Storage.

    Compatible with llama-index framework.

    Taken from wxd-setup-and-ingestion repository in skol-assets

    Attributes
    ----------
    bucket_name : str
        The name of the bucket in the cloud storage.
    credentials : dict
        The credentials required to authenticate with the cloud storage. It must contain 'apikey' and 'service_instance_id'.
    hostname : str, optional
    """

    def __init__(
        self,
        bucket_name: str,
        credentials: dict,
        hostname: str = "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        readers: Optional[Dict[str, BaseReader]] = None,
    ):
        self.bucket_name = bucket_name
        self.credentials = credentials
        self.hostname = hostname
        self._available_readers = readers if readers else {}
        self._base_url = f"{self.hostname}/{self.bucket_name}"
        if "apikey" in self.credentials and "service_instance_id" in self.credentials:
            self.credentials = credentials
        else:
            raise ValueError(
                "Missing 'apikey' or 'service_instance_id' in credentials."
            )
        self._bearer_token = self.__get_bearer_token()

    async def load_data(
        self,
        regex_filter: str = None,
        num_files: int = None,
    ) -> List[Document]:
        async def consume_generator():
            return [
                doc
                async for doc in self.async_load_data(
                    regex_filter=regex_filter, num_files=num_files
                )
            ]

        return await consume_generator()

    async def async_load_data(
        self, regex_filter: str = None, num_files: int = None
    ) -> AsyncGenerator:
        file_names = self.list_files(regex_filter)
        read_tasks = [
            self.read_file_to_documents(file_name)
            for file_name in file_names[:num_files]
        ]
        for read_task in asyncio.as_completed(read_tasks):
            docs = await read_task
            for doc in docs:
                yield doc

    async def read_file_to_documents(self, file_name: str) -> List[Document]:
        file_data = await self.__read_file_data(file_name)
        reader = self.__get_file_reader(file_name)
        file_extension = "." + file_name.split(".")[-1]
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=file_extension
        ) as temp_file:
            temp_file.write(file_data)
            temp_file.flush()
            try:
                logger.info(f"Reading file {file_name}...")
                docs = reader.load_data(temp_file.name)
                for subdoc in docs:
                    subdoc.metadata["filename"] = file_name
            except Exception as e:
                logger.error(f"Failed to read {file_name} with {reader} because of {e}")
                docs = []

        return docs

    def list_files(self, regex_filter: str = None) -> List[str]:
        """
        Lists all the files in the bucket.

        This method sends a GET request to the cloud storage service and parses the response to extract the file names.

        Returns
        -------
        list
            A list of file names.
        """

        @self.__refresh_token_on_exception
        def _list_files(regex_filter: str = None) -> List[str]:
            headers = self.__get_request_header()
            response = requests.request("GET", self._base_url, headers=headers)
            data = response.text
            file_names = re.findall(r"<Key>(.*?)</Key>", data)
            if regex_filter:
                regex = re.compile(regex_filter)
                filtered_file_names = [name for name in file_names if regex.match(name)]
                file_names = filtered_file_names
            return file_names

        return _list_files(regex_filter)

    async def __read_file_data(self, file_name: str) -> bytes:
        """
        Reads a file from the bucket.

        This method sends a GET request to the cloud storage service to read the content of the specified file.

        Parameters
        ----------
        file_name : str
            The name of the file to read.

        Returns
        -------
        bytes
            The content of the file.
        """

        @self.__refresh_token_on_exception
        async def _read_file_data() -> bytes:
            headers = self.__get_request_header()
            url = f"{self._base_url}/{file_name}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.read()
                    return data

        return await _read_file_data()

    @classmethod
    def from_service_credentials(
        cls,
        bucket: str,
        service_credentials_path: Path,
        hostname: str = "https://s3.us-south.cloud-object-storage.appdomain.cloud",
    ) -> "CloudObjectStorageReader":
        with open(service_credentials_path, "r") as file:
            cos_auth_dict = json.load(file)
        credentials = {
            "apikey": cos_auth_dict["apikey"],
            "service_instance_id": cos_auth_dict["resource_instance_id"],
        }
        return cls(bucket_name=bucket, credentials=credentials, hostname=hostname)

    def __get_file_reader(self, file_name: str) -> BaseReader:
        file_extension = "." + file_name.split(".")[-1].lower()
        reader_class_name = DEFAULT_READER_NAMES.get(file_extension)

        if reader_class_name is None:
            raise ValueError(f"No reader available for file extension {file_extension}")

        if file_extension not in self._available_readers:
            logger.info(f"Downloading reader {reader_class_name}...")
            reader = download_loader(reader_class_name)()
            self._available_readers[file_extension] = reader

        return self._available_readers[file_extension]

    def __get_request_header(self) -> Dict[str, str]:
        headers = {
            "ibm-service-instance-id": self.credentials["service_instance_id"],
            "Authorization": f"Bearer {self._bearer_token}",
        }
        return headers

    def __get_bearer_token(self) -> str:
        url = "https://iam.cloud.ibm.com/identity/token"
        payload = f"grant_type=urn%3Aibm%3Aparams%3Aoauth%3Agrant-type%3Aapikey&apikey={self.credentials['apikey']}"
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "accept": "application/json",
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        bearer_token = response.json()["access_token"]
        return bearer_token

    def __refresh_token_on_exception(self, func):
        def wrapper(*args, **kwargs):
            for _ in range(2):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException:
                    self._bearer_token = self.__get_bearer_token()
            raise

        return wrapper


class ElserElasticsearchStore(ElasticsearchStore):
    """Elasticsearch vector store."""

    model_id: Optional[str] = None
    pipeline_name: Optional[str] = None

    def __init__(self, *args, **kwargs):
        pipeline_name = kwargs.pop("pipeline_name", None)
        model_id = kwargs.pop("model_id", None)
        super().__init__(*args, **kwargs)
        self.pipeline_name = pipeline_name
        self.model_id = model_id

    async def aquery(
        self,
        query: VectorStoreQuery,
        custom_query: Optional[
            Callable[[Dict, Union[VectorStoreQuery, None]], Dict]
        ] = None,
        es_filter: Optional[List[Dict]] = None,
        use_llama_similarities: bool = False,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Asynchronous query index for top k most similar nodes.
        Args:
            query_embedding (VectorStoreQuery): query embedding
            custom_query: Optional. custom query function that takes in the es query
                        body and returns a modified query body.
                        This can be used to add additional query
                        parameters to the AsyncElasticsearch query.
            es_filter: Optional. AsyncElasticsearch filter to apply to the
                        query. If filter is provided in the query,
                        this filter will be ignored.
        Returns:
            VectorStoreQueryResult: Result of the query.
        Raises:
            Exception: If AsyncElasticsearch query fails.
        """

        es_query = {}

        if query.filters is not None and len(query.filters.legacy_filters()) > 0:
            filter = [_to_elasticsearch_filter(query.filters)]
        else:
            filter = es_filter or []

        if (
            query.mode
            in (
                VectorStoreQueryMode.DEFAULT,
                VectorStoreQueryMode.HYBRID,
            )
            and query.query_embedding is not None
        ):
            query_embedding = cast(List[float], query.query_embedding)
            es_query["knn"] = {
                "filter": filter,
                "field": self.vector_field,
                "query_vector": query_embedding,
                "k": query.similarity_top_k,
                "num_candidates": query.similarity_top_k * 10,
            }

        if query.mode in (
            VectorStoreQueryMode.TEXT_SEARCH,
            VectorStoreQueryMode.HYBRID,
        ):
            es_query["query"] = {
                "bool": {
                    "must": {"match": {self.text_field: {"query": query.query_str}}},
                    "filter": filter,
                }
            }
        if query.mode == VectorStoreQueryMode.SPARSE:
            es_query["text_expansion"] = {
                "ml.tokens": {"model_id": self.model_id, "model_text": query.query_str}
            }

        if query.mode == VectorStoreQueryMode.HYBRID:
            es_query["rank"] = {"rrf": {}}

        if custom_query is not None:
            es_query = custom_query(es_query, query)
            logger.debug(f"Calling custom_query, Query body now: {es_query}")
        async with self.client as client:
            response = await client.search(
                index=self.index_name,
                query=es_query,
                size=query.similarity_top_k,
                _source={"excludes": [self.vector_field]},
            )

        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        hits = response["hits"]["hits"]
        for hit in hits:
            source = hit["_source"]
            metadata = source.get("metadata", None)
            text = source.get(self.text_field, None)
            node_id = hit["_id"]
            node = metadata_dict_to_node(metadata)
            node.text = text
            top_k_nodes.append(node)
            top_k_ids.append(node_id)
            top_k_scores.append(hit.get("_rank", hit["_score"]))

        if query.mode == VectorStoreQueryMode.HYBRID:
            total_rank = sum(top_k_scores)
            top_k_scores = [total_rank - rank / total_rank for rank in top_k_scores]

        if use_llama_similarities:
            top_k_scores = _to_llama_similarities(top_k_scores)

        return VectorStoreQueryResult(
            nodes=top_k_nodes,
            ids=top_k_ids,
            similarities=top_k_scores,
        )

    def add(
        self,
        nodes: List[BaseNode],
        *,
        create_index_if_not_exists: bool = True,
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to Elasticsearch index.
        Args:
            nodes: List of nodes with embeddings.
            create_index_if_not_exists: Optional. Whether to create
                                        the Elasticsearch index if it
                                        doesn't already exist.
                                        Defaults to True.
        Returns:
            List of node IDs that were added to the index.
        Raises:
            ImportError: If elasticsearch['async'] python package is not installed.
            BulkIndexError: If AsyncElasticsearch async_bulk indexing fails.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.async_add(nodes, create_index_if_not_exists=create_index_if_not_exists)
        )

    async def async_add(
        self,
        nodes: List[BaseNode],
        *,
        create_index_if_not_exists: bool = True,
        create_pipeline_if_not_exists: bool = True,
        **add_kwargs: Any,
    ) -> List[str]:
        """Asynchronous method to add nodes to Elasticsearch index.
        Args:
            nodes: List of nodes with embeddings.
            create_index_if_not_exists: Optional. Whether to create
                                        the AsyncElasticsearch index if it
                                        doesn't already exist.
                                        Defaults to True.
        Returns:
            List of node IDs that were added to the index.
        Raises:
            ImportError: If elasticsearch python package is not installed.
            BulkIndexError: If AsyncElasticsearch async_bulk indexing fails.
        """
        try:
            from elasticsearch.helpers import BulkIndexError, async_bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch[async] python package. "
                "Please install it with `pip install 'elasticsearch[async]'`."
            )

        if len(nodes) == 0:
            return []

        if create_index_if_not_exists:
            try:
                dims_length = len(nodes[0].get_embedding())
            except ValueError:
                dims_length = None
            await self._create_index_if_not_exists(
                index_name=self.index_name, dims_length=dims_length
            )

        if create_pipeline_if_not_exists:
            await self._create_pipeline_if_not_exists()

        requests = []
        return_ids = []
        for node in nodes:
            _id = node.node_id if node.node_id else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": _id,
                "_source": self.__format_node_to_elastic_document(node),
                "pipeline": self.pipeline_name,
            }
            requests.append(request)
            return_ids.append(_id)
        try:
            success, failed = await async_bulk(
                self.client, requests, chunk_size=self.batch_size, refresh=True
            )
            logger.debug(f"Added {success} and failed to add {failed} texts to index")

            logger.debug(f"added texts {return_ids} to index")
            return return_ids
        except BulkIndexError as e:
            logger.error(f"Error adding texts: {e}")
            firstError = e.errors[0].get("index", {}).get("error", {})
            logger.error(f"First error reason: {firstError.get('reason')}")
            raise

    async def _create_pipeline_if_not_exists(self) -> None:
        try:
            await self.client.ingest.get_pipeline(id=self.pipeline_name)
            logger.debug(
                f"Pipeline {self.pipeline_name} already exists. Skipping creation."
            )
        except NotFoundError:
            pipeline_settings = {
                "description": "Inference pipeline using ELSER model",
                "processors": [
                    {
                        "inference": {
                            "field_map": {self.text_field: "text_field"},
                            "model_id": self.model_id,
                            "target_field": "ml",
                            "inference_config": {
                                "text_expansion": {"results_field": "tokens"}
                            },
                        }
                    }
                ],
                "version": 1,
            }
            logger.debug(
                f"Creating pipeline {self.pipeline_name} that uses ELSER model"
            )
            await self.client.ingest.put_pipeline(
                id=self.pipeline_name, body=pipeline_settings
            )

    async def _create_index_if_not_exists(
        self, index_name: str, dims_length: Optional[int] = None
    ) -> None:
        """Create the AsyncElasticsearch index if it doesn't already exist.
        Args:
            index_name: Name of the AsyncElasticsearch index to create.
            dims_length: Length of the embedding vectors.
        """
        if await self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists. Skipping creation.")

        else:
            if dims_length is None:
                logger.info("Using ELSER model since dims_length is None")
                index_settings = {
                    "mappings": {
                        "properties": {
                            "ml.tokens": {"type": "rank_features"},
                            self.text_field: {"type": "text"},
                        }
                    }
                }
            else:
                if self.distance_strategy == "COSINE":
                    similarityAlgo = "cosine"
                elif self.distance_strategy == "EUCLIDEAN_DISTANCE":
                    similarityAlgo = "l2_norm"
                elif self.distance_strategy == "DOT_PRODUCT":
                    similarityAlgo = "dot_product"
                else:
                    raise ValueError(
                        f"Similarity {self.distance_strategy} not supported."
                    )

                index_settings = {
                    "mappings": {
                        "properties": {
                            self.vector_field: {
                                "type": "dense_vector",
                                "dims": dims_length,
                                "index": True,
                                "similarity": similarityAlgo,
                            },
                            self.text_field: {"type": "text"},
                            "metadata": {
                                "properties": {
                                    "document_id": {"type": "keyword"},
                                    "doc_id": {"type": "keyword"},
                                    "ref_doc_id": {"type": "keyword"},
                                }
                            },
                        }
                    }
                }

            logger.debug(
                f"Creating index {index_name} with mappings {index_settings['mappings']}"
            )
            await self.client.indices.create(index=index_name, **index_settings)

    def __format_node_to_elastic_document(self, node: TextNode):
        elastic_doc = {
            self.text_field: node.get_content(metadata_mode=MetadataMode.NONE),
            "metadata": node_to_metadata_dict(node, remove_text=True),
        }
        return elastic_doc