from langchain_core.prompts import PromptTemplate

# Template for answering questions using retrieved context
RAG_PROMPT_TEMPLATE = """
You are an AI assistant specialized in movies, helping users find information and recommendations.
Use the following retrieved context from a movie graph database to answer the user's question.
If the context doesn't contain the answer, state that you don't have enough information from the database but try to answer based on your general knowledge if appropriate, clearly indicating the source of your information.
Provide concise and factual answers. Do not make up information not present in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# Template for explaining recommendations (optional)
RECOMMENDATION_EXPLANATION_TEMPLATE = """
A user asked for movies similar to "{input_movie}".
Based on graph database analysis (shared actors, directors, genres) and semantic plot similarity, here are some recommendations:

RECOMMENDATIONS:
{recommendations_list}

Briefly explain why these movies are good recommendations in a friendly and engaging tone. You can mention common themes, actors, or genres if known from the recommendations list provided. Keep it concise (2-3 sentences).

EXPLANATION:
"""

recommendation_explanation_prompt = PromptTemplate(
    template=RECOMMENDATION_EXPLANATION_TEMPLATE,
    input_variables=["input_movie", "recommendations_list"]
)

# Add more specific prompt templates as needed