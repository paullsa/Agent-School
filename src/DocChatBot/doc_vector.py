# === Required Packages ===
# pip install langchain langchain-community pypdf chromadb tiktoken docx2txt

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# === Step 1: Load PDF File ===
# Load the contents of the PDF using PyPDFLoader
pdf_loader = PyPDFLoader('./docs/US Space Policy.pdf')
documents = pdf_loader.load()

# === Step 2: Split Text into Chunks ===
# This helps the vector store manage and search through large documents
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# === Step 3: Generate Embeddings and Create a Vector Store ===
# We use OllamaEmbeddings for local embedding generation (no API key needed)
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Change model if needed
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory='./data'
)
vectordb.persist()  # Save the vector store for future use

# === Step 4: Initialize Local LLM via Ollama ===
llm = ChatOllama(model="gemma3:1b", temperature=0.0)  # You can use other local models too

# === Step 5: Create RetrievalQA Chain ===
# This lets the LLM search the vector DB for relevant context before answering
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True
)

# === Step 6: Ask a Question ===
result = qa_chain("When was the shuttle era?")
print(result['result'])

# === Optionally: See source documents used ===
# for doc in result['source_documents']:
#     print(doc.page_content)
