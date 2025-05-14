# === Required Libraries ===
# pip install langchain pypdf chromadb tiktoken docx2txt

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# === Load PDF Document ===
# This uses PyPDFLoader to extract text from the given PDF file.
pdf_loader = PyPDFLoader('./docs/US Space Policy.pdf')
documents = pdf_loader.load()

# === Initialize Local LLM using Ollama ===
# This uses a locally running model via Ollama (e.g., llama3, mistral, etc.)
# Make sure Ollama is running and the model is available.
llm = ChatOllama(model="gemma3:1b", temperature=0.0)  # You can adjust model and temperature

# === Set up Question Answering (QA) Chain ===
# This chain takes a list of documents and answers questions based on them.
chain = load_qa_chain(llm, verbose=True)

# === Ask a Question ===
query = 'When did Sputnik launch?'
response = chain.run(input_documents=documents, question=query)

# === Output the Answer ===
print(response)
