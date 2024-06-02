import os
from turtle import st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

from app import cool_header, display_chat, user_input

# Load environment variables
load_dotenv()

# Get Google API Key
GOOGLE_API_KEY = os.getenv("API_KEY")

# Check if GOOGLE_API_KEY is loaded, if not raise an error
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Set the PDF directory
pdf_destination = r"D:\Aniket Digital CV\pdf"

# Get all PDF files from the specified directory
pdf_docs = [
    os.path.join(pdf_destination, pdf_file) 
    for pdf_file in os.listdir(pdf_destination) 
    if pdf_file.endswith(".pdf")
]

# Extract text from PDFs
text = ""
for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:  # Ensure the page has text
            text += page_text

print("Wait!!! VectorDB in making")

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=10000)
text_chunks = text_splitter.split_text(text)

# Initialize embeddings and create a vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)
vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

# Save the vector store locally
vector_store.save_local("chatbot")
print("VectorDb made!!")

cool_header()

user_question = st.text_input("Enter your question here")
chats = st.session_state.get('chats', [])

if st.button("Ask", key="ask_button"):
    with st.spinner("Amy is thinking..."):
        if user_question:
            chats = user_input(user_question, chats)
            st.session_state['chats'] = chats
            user_question = ""  # Clear input field after asking

display_chat(chats)
