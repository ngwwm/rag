from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.models.openai import OpenAIChat
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.azure_openai import AzureOpenAIEmbedder
import os
import argparse

def create_document_agent(documents_dir):
    # Verify the directory exists
    if not os.path.exists(documents_dir):
        raise ValueError(f"Directory {documents_dir} does not exist")

    # Read the API key from the environment variable
    if not os.getenv("AZURE_EMBEDDER_OPENAI_API_KEY"):
        raise ValueError("AZURE_EMBEDDER_OPENAI_API_KEY environment variable is not set")

    # Create knowledge base from local text/documents
    knowledge_base = TextKnowledgeBase(
        path=documents_dir,  # Directory containing .txt or .docx files
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="documents",
            search_type=SearchType.hybrid,
            embedder=AzureOpenAIEmbedder(
                azure_endpoint="https://marti-m7i61trw-eastus2.cognitiveservices.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15",
                id="text-embedding-3-small"
            )
        )
    )

    # Read the API key from the environment variable
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

    # Create the agent
    agent = Agent(
        model=AzureOpenAI(
            azure_endpoint="https://mywebapp-openai.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
            id="gpt-4o-mini"),
        description="An agent that reads and summarizes documents from a local directory",
        instructions=[
            "Read and understand all documents in the knowledge base",
            "Provide concise summaries when asked",
            "Answer questions based on the document contents",
            "If information is not found in documents, say so clearly"
        ],
        knowledge=knowledge_base,
        markdown=True
    )

    # Load the knowledge base (comment out after first run if unchanged)
    if agent.knowledge is not None:
        print("Loading documents into knowledge base...")
        agent.knowledge.load()
        print("Documents loaded successfully")

    return agent

def main():
    parser = argparse.ArgumentParser(description="Create a document agent.")
    parser.add_argument("documents_dir", type=str, help="The directory containing the documents.")
    args = parser.parse_args()

    try:
        # Initialize the agent
        agent = create_document_agent(args.documents_dir)

        # Example usage
        
        # Request a summary
        print("\nGenerating summary of all documents:")
        agent.print_response("Please provide a summary of all documents in the directory")

        # Ask a specific question
        print("\nAnswering a sample question:")
        agent.print_response("What is the main topic discussed in the documents?")

        # Interactive question loop
        while True:
            question = input("\nAsk a question about the documents (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            agent.print_response(question)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Make sure to set your OpenAI API key as an environment variable
    # export AZURE_OPENAI_API_KEY='your-api-key-here'
    
    main()
