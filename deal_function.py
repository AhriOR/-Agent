from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from my_agent import agent_executor


#模型初始化
deepseek_KEY = 'sk-6321ebc75a0741e68625b92c48cb6e61'
deepseek_URL = 'https://api.deepseek.com'
llm = ChatOpenAI(
    openai_api_key=deepseek_KEY,
    openai_api_base=deepseek_URL,
    model_name='deepseek-chat',
    streaming=False,
    temperature=0.7
)

#加载text
def txt_loader(file_path,user_id):
    with open(file_path,'r',encoding='utf-8') as paper:
        file=paper.read()
        file = re.sub(r'\n{2,}', '\n', file)
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        length_function=len,
        add_start_index=True
    )
    text=text_splitter.create_documents([file])

    document=[]
    for idx,chunk in enumerate(text):
        metadata={
            'source': file_path,
            'id':idx,
            'user_id':user_id,
            **chunk.metadata
        }
        document.append(Document(page_content=chunk.page_content,metadata=metadata))
    return document

#加载pdf
def pdf_loader(file_path,user_id):
    loader=PyPDFLoader(file_path)
    pages=loader.load()
    documents = []
    for idx, page in enumerate(pages):
        # 对 page_content 进行清洗
        cleaned_text = re.sub(r'\s*\n\s*', '\n', page.page_content)  # 压缩空行
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)  # 多空格替换为单空格
        cleaned_text = cleaned_text.strip()  # 去除首尾空格

        # 组合metadata，可以加id和source等
        metadata = {
            'source': file_path,
            'id': idx,
            'user_id':user_id,
            **page.metadata  # 保留原metadata
        }
        # 生成新的 Document
        documents.append(Document(page_content=cleaned_text, metadata=metadata))


    return documents

def agent_response(text_input):

    result=agent_executor.invoke({'input':text_input})
    return result['output']





