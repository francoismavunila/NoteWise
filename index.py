from langchain_community.document_loaders import PyPDFLoader
import asyncio

# lets first just read the doc
file_path = (
    "./docs/Reality_is_Broken.pdf"
)

loader = PyPDFLoader(file_path)
# async def read_pdf():
for page in loader.lazy_load():
    print(page)
    print('-----')
    
# asyncio.run(read_pdf())