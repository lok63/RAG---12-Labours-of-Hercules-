from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from src.config import OPENAI_KEY


def get_text_embedding_3_large() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key= OPENAI_KEY
    )


def get_huggingface_embeddings(model_name: str="sentence-transformers/all-mpnet-base-v2", model_kwargs :dict = {'device': 'cpu'}, encode_kwargs:dict = {'normalize_embeddings': False} )-> HuggingFaceHubEmbeddings:
    return HuggingFaceHubEmbeddings(
        model_name =model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )



