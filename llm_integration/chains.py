from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm_integration.llm_config import get_llm
from llm_integration.prompts import rag_prompt
import logging

log = logging.getLogger(__name__)

def format_context(context_dict: dict) -> str:
    """Formats the retrieved context dictionary into a string for the LLM prompt."""
    if not context_dict:
        return "No context available."

    formatted_context = f"Movie: {context_dict.get('title', 'N/A')}\n"
    formatted_context += f"Released: {context_dict.get('released', 'N/A')}\n"
    formatted_context += f"Tagline: {context_dict.get('tagline', 'N/A')}\n"
    # Keep plot separate or summarize if too long
    # formatted_context += f"Plot Summary: {context_dict.get('plot', 'N/A')}\n"
    formatted_context += f"Director(s): {', '.join(context_dict.get('directors', ['N/A']))}\n"
    formatted_context += f"Actor(s): {', '.join(context_dict.get('actors', ['N/A']))}\n"
    formatted_context += f"Genre(s): {', '.join(context_dict.get('genres', ['N/A']))}\n"
    return formatted_context

def create_rag_chain():
    """Creates the main RAG chain using LangChain Expression Language (LCEL)."""
    try:
        llm = get_llm()
    except ValueError as e:
        log.error(f"Cannot create RAG chain: {e}")
        return None # Indicate failure

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": (lambda x: format_context(x["context"])), "question": RunnablePassthrough()} # Format context before passing to prompt
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    # Note: The input to this chain should be a dictionary:
    # {'context': {'title': ..., 'actors': ...}, 'question': 'User question here'}

    return rag_chain

# Optional: Chain for explaining recommendations
def create_explanation_chain():
    try:
        llm = get_llm()
    except ValueError:
        return None
    explanation_chain = (
        recommendation_explanation_prompt
        | llm
        | StrOutputParser()
    )
    # Input: {'input_movie': '...', 'recommendations_list': '...'}
    return explanation_chain

#Example Usage (Conceptual)
if __name__ == "__main__":
    rag_chain = create_rag_chain()
    if rag_chain:
        # Dummy context and question
        dummy_context = {
            'title': 'Inception',
            'released': 2010,
            'tagline': 'Your mind is the scene of the crime.',
            'plot': 'A thief who steals corporate secrets through the use of dream-sharing technology...',
            'directors': ['Christopher Nolan'],
            'actors': ['Leonardo DiCaprio', 'Joseph Gordon-Levitt', 'Elliot Page'],
            'genres': ['Action', 'Sci-Fi', 'Thriller']
        }
        question = "Who directed Inception and what year was it released?"

        # Invoke the chain
        response = rag_chain.invoke({"context": dummy_context, "question": question})
        print("RAG Chain Response:")
        print(response)
    else:
        print("Could not create RAG chain. Check LLM configuration.")