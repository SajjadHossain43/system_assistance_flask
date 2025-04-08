from flask import Flask, render_template, jsonify, request
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['GET'])
def get_system_response() -> str:
    user_input = request.args.get('user_input')
    #return jsonify({"message": f"{process_input(user_input, f'{os.getcwd()}/FAISS')}"})
    return process_input(user_input, f'{os.getcwd()}/FAISS')

def get_system_response(question: str) -> str:
    return process_input(question, f'{os.getcwd()}/FAISS')

def auto_config():
    os.environ["API_KEY"] = "2EqAWpytfgNV4iFEvQ1suZUdAQx142L346bnKLpcGmoyv1tfJOSWJQQJ99BCACHrzpqXJ3w3AAAAACOG3yRc"
    os.environ["GPT_API_VERSION"] = "2025-01-01-preview" 
    os.environ["EMBED_API_VERSION"] = "2023-05-15"
    os.environ["GPT_MODEL"] ="gpt-35-turbo-16k"
    os.environ["EMBED_MODEL"] ="text-embedding-3-large"
    os.environ["OPENAI_ENDPOINT"] ="https://hight-m87lalwz-northcentralus.cognitiveservices.azure.com/"
    
def get_gpt_instance():
    return AzureChatOpenAI(azure_deployment=os.getenv('GPT_MODEL'), api_key=os.getenv('API_KEY'), api_version=os.getenv('GPT_API_VERSION'), azure_endpoint=os.getenv('OPENAI_ENDPOINT'))

def get_embedding_instance():
    return AzureOpenAIEmbeddings(azure_deployment=os.getenv('EMBED_MODEL'), api_key=os.getenv('API_KEY'), api_version=os.getenv('EMBED_API_VERSION'), azure_endpoint=os.getenv('OPENAI_ENDPOINT'))

def get_text_from_pdf(pdf_path: os.path) -> str:
    texts = ''
    if not os.path.exists(pdf_path): return texts
    pdfs = glob.glob(os.path.join(pdf_path, '*.pdf'))
    if not pdfs: return texts
    for pdf in pdfs:
        with open(pdf, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if len(text) > 0: texts += text
    return texts

def get_chunk_from_texts(texts: str):
    return RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(texts)

def generate_index(pdf_path: os.path, index_path: os.path):
    if not os.path.exists(os.path.join(pdf_path, 'index.faiss')):
        texts = get_text_from_pdf(pdf_path)
        if len(texts) > 0:
            text_chunks = get_chunk_from_texts(texts)
            embedding = get_embedding_instance()
            vectors = FAISS.from_texts(text_chunks, embedding=embedding)
            vectors.save_local(index_path)
    return True

def get_conversation_chain():
    prompt_template = """
     This is a Q&A prompt for TribaLex, a court solution system. Respond concisely. Try to talk in more human-like form. 
     - If you do not find the answer in the context, say  you don't know politely . 
     - Do not generate any information outside of the context.
     - If you're unsure or the context does not have enough information, simply say you are sorry, you don't have enough information to answer that, in a polite manner.
     - Elaborate on the answer with proper information.
     Context: \n {context}? \n
     Question: \n {question}? \n
     Answer:
     """""
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    llm = get_gpt_instance()
    llm.model_rebuild()
    return load_qa_chain(llm=llm, chain_type='stuff', prompt=prompt)

def process_input(question: str, index_path: os.path) -> str:
    auto_config()
    print("GPT_MODEL:", os.getenv("GPT_MODEL"))
    print("API_KEY:", os.getenv("API_KEY"))
    print("GPT_API_VERSION:", os.getenv("GPT_API_VERSION"))
    print("OPENAI_ENDPOINT:", os.getenv("OPENAI_ENDPOINT"))
    pdf_content = FAISS.load_local(index_path, embeddings=get_embedding_instance(), allow_dangerous_deserialization=True)
    if len(question) > 0:
        model_input = pdf_content.similarity_search(question)
    if model_input:
        try:
            chain = get_conversation_chain()
            output = chain({'input_documents': model_input, 'question': question}, return_only_outputs=True)['output_text']
            #return json.dumps({'answer': output})
            return output
        except Exception as e:
            #return json.dumps({'answer': 'Sorry, I am not able to answer your question due to some problem'})
            return e.with_traceback()

#def save_file(file: UploadFile = File(...)):
    #if not os.path.exists(os.path.join(os.getcwd(), 'pdf')): os.mkdir(f'{os.getcwd()}/pdf')
    #if not os.path.exists(os.path.join(os.getcwd(), 'FAISS')): os.mkdir(f'{os.getcwd()}/FAISS')
    #file_path = f"{os.path.join(os.getcwd(), 'pdf')}/1.pdf"
    #with open(file_path, "wb") as f:
        #f.write(await file.read())
    #generate_index(pdf_path=os.path.join(os.getcwd(), 'pdf'), index_path=os.path.join(os.getcwd(), 'FAISS'))
    #return f"File saved as {file_path}"

if __name__ == '__main__':
    app.run(debug=True)