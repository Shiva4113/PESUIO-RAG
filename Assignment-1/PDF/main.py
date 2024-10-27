#IMPORTS
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.cerebras import Cerebras
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


load_dotenv()

#API KEYS
JINA_API_KEY =  os.getenv('JINA_API_KEY')
CEREBRAS_API_KEY =  os.getenv('CEREBRAS_API_KEY')


#CONGIG
def configure():
    global Settings
    Settings.embed_model = JinaEmbedding(
        api_key=JINA_API_KEY,
        model='jina-embeddings-v3',
        task='retrieval.passage',
        embed_batch_size=16
    )

    Settings.llm = Cerebras(
        api_key=CEREBRAS_API_KEY,
        model="llama3.1-70b"
    )


def create_rag_system(data_dir="./data"):
    client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_documents",
        dimension=1024
    )
    documents = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store
    )
    query_engine = index.as_query_engine()
    return query_engine

def query_rag(query_engine, question: str):
    response = query_engine.query(question)
    return response


def main():
    configure()
    query_engine = create_rag_system()
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        response = query_rag(query_engine, question)
        print(f"\nAnswer: {response}")
    
if __name__ == "__main__":
    main()