import langchain
from langchain.chains import ConversationChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate

# Initialize your FAISS vector store with preloaded documents
def load_faiss_store():
    # Example: Load or create your FAISS vector store (ensure your documents are embedded)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    faiss_index = FAISS.load_local("C:\Python\Agent-School\src\faiss_index_docs2", embeddings)
    return faiss_index

# Initialize Wikipedia tool
wiki_api = WikipediaAPIWrapper(top_k_results=1, lang="en")
wikipedia_tool = WikipediaQueryRun(api_wrapper=wiki_api)

# Define custom tools with a condition for selecting between Wikipedia or FAISS
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="Use Wikipedia to find information",
    ),
    Tool(
        name="FAISS",
        func=lambda query: faiss_index.similarity_search(query, k=5),
        description="Use FAISS to retrieve information from a vector store",
    )
]

# Define a simple condition-based function to choose the tool
def decide_tool(query):
    # Define your own conditions for deciding the tool
    if len(query.split()) < 5:  # e.g., short queries go to Wikipedia
        return "Wikipedia"
    else:  # longer queries use FAISS for potentially more context
        return "FAISS"

# Define agent to decide on the tool dynamically
class CustomAgent:
    def __init__(self, tools):
        self.tools = tools
    
    def run(self, query):
        chosen_tool_name = decide_tool(query)
        chosen_tool = next(tool for tool in self.tools if tool.name == chosen_tool_name)
        return chosen_tool.func(query)

# Initialize FAISS vector store
faiss_index = load_faiss_store()

# Create the agent
agent = CustomAgent(tools)

# Example query
query = "What is the capital of France?"

# Get the result
result = agent.run(query)
print(result)
