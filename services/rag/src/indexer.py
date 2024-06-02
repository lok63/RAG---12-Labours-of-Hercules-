from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader

from src.config import HERCULES_TRAINING_SET
from src.embedding_models import get_text_embedding_3_large
from src.llm_models import get_chat_gpt_3_5
from src.prompts.qa_prompt import qa_prompt_no_memory


def load_data_from_csv_into_pandas(
    path: str, source_column: str | None = None, metadata_columns: List[str] = None
) -> pd.DataFrame:
    df = pd.read_csv(path)

    data = loader.load()
    return data


def load_data_from_csv(
    path: str, source_column: str | None = None, metadata_columns: List[str] = None
) -> List[Document]:
    loader = CSVLoader(
        file_path=path,
        source_column=source_column,
        metadata_columns=metadata_columns,
        csv_args={'fieldnames': ['Question', 'Ground Truth']},
    )

    data = loader.load()
    return data


def load_data_from_txt(path: str) -> List[Document]:
    loader = TextLoader(path)
    data = loader.load()
    return data


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return 'No context'

    output = ""
    for i, doc in enumerate(docs):
        output += f"Sample {i+1}\n"
        output += f"Textbook Content: {doc.page_content}\n\n"
    return output


if __name__ == '__main__':
    embeddings = get_text_embedding_3_large()
    llm = get_chat_gpt_3_5()
    # docs = load_data_from_csv(str(HERCULES_VALIDATION_SET), 'Question', ['Ground Truth'])
    docs = load_data_from_txt(str(HERCULES_TRAINING_SET))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 2,
            "score_threshold": 0.5,
        },
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | qa_prompt_no_memory
        | llm
        | StrOutputParser()
    )

    res = rag_chain.invoke("What is Task Decomposition?")
    print(res)
    # cleanup
    vectorstore.delete_collection()
