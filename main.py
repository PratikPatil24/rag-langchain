import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough


load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving Data from Vector DB")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o")

    query = "what is Pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm

    res = chain.invoke({"input": None})
    print("\n\n---------------------------------")
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
    print("\n\n---------------------------------")
    print(result)

    # Retrieval Implementation with LCEL

    template = """
        Use the following pieces od context to answer the question at the end. If you don't know the answer, 
        just say that you don't know, don't try to make an answer. Use three sentences maximum and keep the answer as concise as possible.
        Always say "Thanks for asking!" at the end of the answer.
        
        {context}
        
        Question: {question}
        Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate(template=template)

    rag_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | custom_rag_prompt
        | llm
    )

    response = rag_chain.invoke(query)
    print("\n\n---------------------------------")
    print(response)
