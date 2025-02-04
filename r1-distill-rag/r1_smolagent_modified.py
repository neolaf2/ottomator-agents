from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import Optional

load_dotenv()

class LLMConfig:
    def __init__(self, provider=None, model=None):
        self.provider = (provider or "huggingface").lower()  # Provide default if None
        self.model = model
        
    def get_api_config(self) -> tuple[str, str, Optional[str]]:
        """Returns (api_base, api_key, org_id) for the provider"""
        if self.provider == "openai":
            return (
                os.getenv("OPENAI_API_BASE"),
                os.getenv("OPENAI_API_KEY"),
                os.getenv("OPENAI_ORG_ID")
            )
        elif self.provider == "deepseek":
            return (
                os.getenv("DEEPSEEK_API_BASE"),
                os.getenv("DEEPSEEK_API_KEY"),
                None
            )
        elif self.provider == "anthropic":
            return (
                os.getenv("ANTHROPIC_API_BASE"),
                os.getenv("ANTHROPIC_API_KEY"),
                None
            )
        elif self.provider == "google":
            return (
                os.getenv("GOOGLE_API_BASE"),
                os.getenv("GOOGLE_API_KEY"),
                None
            )
        elif self.provider == "groq":
            return (
                os.getenv("GROQ_API_BASE"),
                os.getenv("GROQ_API_KEY"),
                None
            )
        elif self.provider == "ollama":
            return (
                os.getenv("OLLAMA_API_BASE"),
                "ollama",
                None
            )
        elif self.provider == "lmstudio":
            return (
                os.getenv("LMSTUDIO_API_BASE"),
                "lmstudio",
                None
            )
        elif self.provider == "huggingface":
            return (
                os.getenv("HUGGINGFACE_API_BASE"),
                os.getenv("HUGGINGFACE_API_TOKEN"),
                None
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

def get_model(config: LLMConfig):
    """Create a model instance based on the provider configuration"""
    if config.provider == "huggingface":
        return HfApiModel(
            model_id=config.model,
            token=config.get_api_config()[1],
            model_kwargs={"model": config.model}
        )
    else:
        api_base, api_key, org_id = config.get_api_config()
        model_kwargs = {"api_base": api_base, "api_key": api_key}
        if org_id:
            model_kwargs["organization"] = org_id
        return OpenAIServerModel(
            model_id=config.model,
            **model_kwargs
        )

# Configure the agents
primary_config = LLMConfig(
    provider=os.getenv("PRIMARY_AGENT_PROVIDER"),
    model=os.getenv("PRIMARY_AGENT_MODEL")
)

reason_config = LLMConfig(
    provider=os.getenv("REASON_AGENT_PROVIDER"),
    model=os.getenv("REASON_AGENT_MODEL")
)

# Create the models
reasoning_model = get_model(reason_config)
tool_model = get_model(primary_config)

# Create the agents
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)

# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

@tool
def rag_with_reasoner(user_query: str) -> str:
    """
    This is a RAG tool that takes in a user query and searches for relevant content from the vector database.
    The result of the search is given to a reasoning LLM to generate a response, so what you'll get back
    from this tool is a short answer to the user's question based on RAG context.

    Args:
        user_query: The user's question to query the vector database with.
    """
    # Search for relevant documents
    docs = vectordb.similarity_search(user_query, k=3)
    
    # Combine document contents
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # Create prompt with context
    prompt = f"""Based on the following context, answer the user's question. Be concise and specific.
    If there isn't sufficient information, give as your answer a better query to perform RAG with.
    
Context:
{context}

Question: {user_query}

Answer:"""
    
    # Get response from reasoning model
    response = reasoner.run(prompt, reset=False)
    return response

# Create the primary agent to direct the conversation
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner], model=tool_model, add_base_tools=False, max_steps=3)

# Example prompt: Compare and contrast the services offered by RankBoost and Omni Marketing
def main():
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    main()