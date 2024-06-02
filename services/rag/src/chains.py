from typing import Tuple

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.config import HERCULES_TRAINING_SET
from src.embedding_models import get_text_embedding_3_large
from src.indexer import load_data_from_txt, format_docs
from src.llm_models import get_chat_gpt_3_5
from src.prompts.qa_prompt import qa_prompt_no_memory


def get_in_memory_retriever_qa_chain() -> Tuple[Runnable, Chroma]:
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
            "score_threshold": 0.2,
        },
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | qa_prompt_no_memory
    #     | llm
    #     | StrOutputParser()
    # )

    return rag_chain, vectorstore


if __name__ == '__main__':
    chain, vectorstore = get_in_memory_retriever_qa_chain()
    input_ = 'who was Hercules father?'
    res = chain.invoke(input_)
    print(res)
    print(f"Answer:\n {res['result']}")
    vectorstore.delete_collection()
