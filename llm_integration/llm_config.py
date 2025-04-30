from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic # Example for Anthropic
# from langchain_community.llms import Ollama # Example for local Ollama
from config import settings
import logging

log = logging.getLogger(__name__)

def get_llm():
    """Initializes and returns the configured LLM instance."""
    # --- OpenAI Example ---
    if settings.OPENAI_API_KEY:
        log.info("Initializing OpenAI LLM.")
        try:
            llm = ChatOpenAI(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.OPENAI_MODEL_NAME if hasattr(settings, 'OPENAI_MODEL_NAME') else "gpt-3.5-turbo", # Default model
                temperature=0.3 # Adjust temperature for creativity vs. factuality
            )
            return llm
        except Exception as e:
            log.error(f"Failed to initialize OpenAI LLM: {e}")
            # Fall through to try other providers or raise error

    # --- Add other LLM providers here ---
    # Example: Anthropic
    # if settings.ANTHROPIC_API_KEY:
    #     log.info("Initializing Anthropic LLM.")
    #     try:
    #         llm = ChatAnthropic(anthropic_api_key=settings.ANTHROPIC_API_KEY, model_name="claude-3-sonnet-20240229")
    #         return llm
    #     except Exception as e:
    #         log.error(f"Failed to initialize Anthropic LLM: {e}")

    # --- Example: Local Ollama ---
    # try:
    #     log.info("Attempting to initialize Ollama LLM.")
    #     # Assumes Ollama server is running locally
    #     llm = Ollama(model="llama3") # Specify your Ollama model
    #     # You might add a quick check here to see if it connects
    #     # llm.invoke("Hi")
    #     log.info("Ollama LLM initialized.")
    #     return llm
    # except Exception as e:
    #     log.info(f"Could not initialize Ollama LLM (is server running?): {e}")


    # If no LLM could be initialized
    log.error("No suitable LLM provider configured or initialization failed.")
    raise ValueError("LLM could not be initialized. Please check configuration (.env file).")

# Example usage
# if __name__ == "__main__":
#     try:
#         llm_instance = get_llm()
#         print(f"LLM Initialized: {type(llm_instance)}")
#         # Example invocation (requires API key to be set)
#         # response = llm_instance.invoke("Explain what a graph database is in one sentence.")
#         # print(response.content)
#     except ValueError as e:
#         print(e)
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")