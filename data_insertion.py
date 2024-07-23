import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == "__main__":
    print("Data Insertion")
    loader = TextLoader(
        "/home/pratik-patil/Learnings/Langchain/rag-langchain/blogs/mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings_model = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(
        texts,
        embeddings_model,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
    print("Data Insertion Complete!")
