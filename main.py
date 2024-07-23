import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv()


if __name__ == "__main__":
    print("Retrieving Data from Vector DB")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o")

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm

    res = chain.invoke({"input": None})
    print(res)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    print(retrieval_prompt)

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke({"input": query})
    print(result)
