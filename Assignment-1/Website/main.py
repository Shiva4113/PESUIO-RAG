import requests
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

def create_md():
    for i in range(1,11):

        url = f'https://r.jina.ai/https://www.lightnovelworld.co/novel/the-beginning-after-the-end-548/chapter-{i}-30041322'
        response = requests.get(url)

        if response.status_code == 200:

            content = response.text
            start = content.find("Markdown Content:")
            end = content.find("Default Dyslexic Roboto Lora")
            
            if start != -1 and end != -1:
                cleaned_content = content[start + len("Markdown Content:"):end].strip()

                with open('./data/temp.md', 'a', encoding='utf-8') as file:
                    file.write(cleaned_content)
                
                print("Content saved successfully.")
            else:
                print("Could not find the specified sections in the content.")
        else:
            print("Failed to retrieve content. Status code:", response.status_code)

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
    create_md()
    query_engine = create_rag_system()
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        response = query_rag(query_engine, question)
        print(f"\nAnswer: {response}")
    
if __name__ == "__main__":
    main()
