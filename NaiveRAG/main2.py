import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.cerebras import Cerebras

class SimpleRAG:
    def __init__(self, pdf_path: str):
        # Load API keys
        load_dotenv()
        self.cerebras_key = os.getenv('CEREBRAS_API_KEY')
        self.jina_key = os.getenv('JINA_API_KEY')
        
        if not self.cerebras_key or not self.jina_key:
            raise ValueError("API keys not found in .env file")
        
        # Initialize settings and create index
        self._setup_settings()
        self.index = self._create_index(pdf_path)
    
    def _setup_settings(self):
        """Configure global settings for LlamaIndex"""
        Settings.llm = Cerebras(
            api_key=self.cerebras_key,
            model="llama3.1-70b"
        )
        
        Settings.embed_model = JinaEmbedding(
            api_key=self.jina_key,
            model_name="jina-embeddings-v2-base-en"
        )
    
    def _create_index(self, pdf_path):
        """Create vector index from PDF document"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Load and index the PDF
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        documents = reader.load_data()
        return VectorStoreIndex.from_documents(documents)
    
    def query(self, question: str):
        """Query the RAG system"""
        query_engine = self.index.as_query_engine()
        return query_engine.query(question)

def main():
    # Example usage
    pdf_path = "./data/dataset_test.pdf"
    

    rag = SimpleRAG(pdf_path)
    
    # Example question
    question = "What dietary recommendations are there for managing type 2 diabetes?"
    response = rag.query(question)
    print(f"Question: {question}")
    print(f"Answer: {response}")
        

if __name__ == "__main__":
    main()