import os
import pytesseract
import streamlit as st
import time

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pdf2image import convert_from_path


load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

st.title("Talk to your Documents")
st.sidebar.title("Model fine-tuning")
model=st.sidebar.selectbox("Select the model that you want to use", 
                     ('llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'llama3-70b-8192', 
                      'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma2-9b-it', 'gemma-7b-it'))
temprature=st.sidebar.slider("Temprature", min_value=0., max_value=1., value=0.7)
tokens=st.sidebar.slider("Tokens", min_value=256, max_value=4096, value=1024)

llm=ChatGroq(groq_api_key=groq_api_key, model_name=model, temperature=temprature, max_tokens=tokens)
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
Be respectful and friendly and you can use emojis too   
You do not know anything out of context and if the question 
is out of context simply say that you do not know.
Do not provide output based on your general knowledge
The response provied must be more than 256 tokens.
<context>
{context}
<context>
Questions:{input}
"""
)

# from transformers import QwenVLProcessor, QwenVLForVisionLanguage
# from pdf2image import convert_from_path

# # Load Qwen2-VL-7B model and processor
# processor = QwenVLProcessor.from_pretrained("Qwen/Qwen2-VL-7B")
# model = QwenVLForVisionLanguage.from_pretrained("Qwen/Qwen2-VL-7B")

# def ocr_pdf_page(file_path, page_number):
#     # Convert PDF page to an image
#     images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
#     if images:
#         # Process the image through Qwen2-VL-7B
#         inputs = processor(images[0], return_tensors="pt")
#         outputs = model(**inputs)
#         ocr_text = outputs.text
#         return ocr_text
#     return ""

def process_pdf_documents(docs):
    for doc in docs:
        if not doc.page_content.strip():
            ocr_text = ocr_pdf_page(doc.metadata['source'], doc.metadata['page'])
            doc.page_content = ocr_text
    return docs

def ocr_pdf_page(file_path, page_number):
    images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
    if images:
        return pytesseract.image_to_string(images[0])
    return ""

def process_pdf_documents(docs):
    for doc in docs:
        if not doc.page_content.strip():
            ocr_text = ocr_pdf_page(doc.metadata['source'], doc.metadata['page'])
            doc.page_content = ocr_text
    return docs

def load_and_process_csv_files(directory_path):
    documents = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            csv_loader = CSVLoader(file_path)
            csv_docs = csv_loader.load()
            for csv_doc in csv_docs:
                documents.append(csv_doc)
    return documents

def vector_embedding_pdfs():
    if "vector" not in st.session_state:
        ## Embeddings: HF / Ollama
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        # st.session_state.embeddings=OllamaEmbeddings(model="all-minilm")
        ## PDF Documents loader
        st.session_state.pdf_loader=PyPDFDirectoryLoader("./pdfFile")
        st.session_state.pdf_docs=process_pdf_documents(st.session_state.pdf_loader.load())
        ## CSV Documents loader
        st.session_state.csv_docs=load_and_process_csv_files("./csvFiles")  
        ## Final Documents ==> PDF + CSV
        st.session_state.final_docs=st.session_state.pdf_docs + st.session_state.csv_docs
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.final_docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
        st.session_state.vectors.save_local("./faissDsRagGroq", "index_hf")

if st.button("Documents Embedding"):
    start=time.process_time()
    vector_embedding_pdfs()
    end=time.process_time()
    st.write("Embedding completed!!!")
    st.write(f"Time taken for generating embeddings: {(end - start):.2f} seconds...")

if st.button("Load vector db"):
    ## Embeddings: HF / Ollama
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    # st.session_state.embeddings=OllamaEmbeddings(model="all-minilm")

    start=time.process_time()
    ## Load vector DB according to the embeddings used: HF / Ollama
    # st.session_state.vector_db = FAISS.load_local("./faissDsRagGroq", embeddings=st.session_state.embeddings,
    #                                                allow_dangerous_deserialization=True)
    st.session_state.vector_db = FAISS.load_local("./faissDsRagGroq", embeddings=st.session_state.embeddings,
                                                   index_name="index_hf" , allow_dangerous_deserialization=True)
    
    end=time.process_time()
    st.write("Embeddings Loaded!!!")
    st.write(f"Time taken for loading embeddings: {(end - start):.2f} seconds")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

input_prompt = st.text_input("Enter Your Question From Documents")

if input_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": input_prompt})
    response_time = time.process_time() - start

    # Store the question and response in chat history
    st.session_state.chat_history.append({"question": input_prompt, "response": response['answer']})

    st.write(f"Response time: {response_time:.2f} seconds")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

    # Display chat history
    with st.expander("Chat History"):
        for chat in st.session_state.chat_history:
            st.write(f"**Question:** {chat['question']}")
            st.write(f"**Response:** {chat['response']}")
            st.write("---")