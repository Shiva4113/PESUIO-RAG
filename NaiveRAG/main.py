import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

LLAMACLOUD_API_KEY = os.getenv('LLAMACLOUD_API_KEY')

parser = LlamaParse(result_type='markdown')

file_extractor = {'.pdf':parser}

output_doc = SimpleDirectoryReader(input_files=['./data/UE18CS303_ML_Unit1.pdf'])
docs = output_doc.load_data()
md_text = ""
for doc in docs:
    md_text += doc.text

with open("output.md") as file_handle:
    file_handle.write(md_text)

print("MD file created successfully")