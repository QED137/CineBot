# core/cypher_query_generation.py

import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from langchain_neo4j import Neo4jGraph # If executing Cypher here

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Schema and Few-Shot Examples (as before) ---
NEO4J_MOVIE_SCHEMA_FOR_LLM = """
# ... (your full schema description) ...
The Neo4j graph contains information about movies.
Node labels and their relevant properties:
- Movie: tmdb_id (STRING), title (STRING), overview (STRING), tagline (STRING), release_date (STRING 'YYYY-MM-DD'), vote_average (FLOAT), poster_url (STRING), trailer_url (STRING)
- Person: tmdb_id (STRING), name (STRING)
- Genre: name (STRING 'Action', 'Sci-Fi', etc.)
Relationships:
- (:Person)-[:DIRECTED]->(:Movie)
- (:Person)-[:ACTED_IN {roles: LIST<STRING>}]->(:Movie)
- (:Movie)-[:HAS_GENRE]->(:Genre)
Instructions for Cypher Generation: Return useful movie properties. Limit results. Handle dates.
"""
FEW_SHOT_EXAMPLES_FOR_LLM = """
# ... (your few-shot examples) ...
User Query: "Movies directed by Christopher Nolan"
Cypher Query: MATCH (p:Person {name: 'Christopher Nolan'})-[:DIRECTED]->(m:Movie) RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview ORDER BY m.release_date DESC LIMIT 15
"""

# --- Core Text-to-Cypher Generation Function ---
def generate_cypher_query_from_natural_language(
    user_query: str,
    openai_client: OpenAI, # Expects an initialized OpenAI client
    llm_model: str = "gpt-3.5-turbo-0125"
) -> Optional[str]:
    # ... (implementation from the previous response - no changes needed here) ...
    if not openai_client:
        logger.error("OpenAI client not provided to generate_cypher_query.")
        return None
    prompt = f"""You are an expert Neo4j Cypher query generator...
Schema:
{NEO4J_MOVIE_SCHEMA_FOR_LLM}
Examples:
{FEW_SHOT_EXAMPLES_FOR_LLM}
User Query: "{user_query}"
Cypher Query:"""
    try:
        completion = openai_client.chat.completions.create(
            model=llm_model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=350
        )
        generated_cypher = completion.choices[0].message.content.strip()
        if "QUERY_NOT_FEASIBLE" in generated_cypher.upper() or not generated_cypher.strip().upper().startswith("MATCH"):
            return None
        return generated_cypher
    except Exception as e:
        logger.error(f"Error generating Cypher from LLM: {e}", exc_info=True)
        return None

# --- Function to Execute Cypher (Now within this module) ---
def execute_generated_cypher(
    cypher_query: str,
    neo4j_graph_instance: Neo4jGraph # Expects an initialized Neo4jGraph instance
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    """
    Executes a given Cypher query using the provided Neo4jGraph instance.
    Returns (results_list, user_friendly_error_message).
    """
    if not neo4j_graph_instance:
        logger.error("Neo4jGraph instance not provided to execute_generated_cypher.")
        return None, "Database connection is not available."
    if not cypher_query:
        logger.warning("No Cypher query provided to execute.")
        return None, "No database query was formed."
    try:
        logger.info(f"Executing Cypher (from cypher_query_generation.py): {cypher_query}")
        results = neo4j_graph_instance.query(cypher_query)
        logger.info(f"Cypher query executed, found {len(results) if results else 0} results.")
        return results if results else [], None
    except Exception as e:
        logger.error(f"Error executing generated Cypher query '{cypher_query}': {e}", exc_info=True)
        err_str = str(e).lower()
        if "syntax error" in err_str: return None, "Query formation error (syntax). Try rephrasing."
        elif "label" in err_str or "property" in err_str: return None, "Couldn't find specific info (property/type). Try rephrasing."
        else: return None, "Database query error. Please try again."

# --- Top-Level Orchestrator Function (Now within this module) ---
def answer_graph_query_via_llm(
    user_query: str,
    openai_client: OpenAI,        # Pass initialized clients
    neo4j_graph_instance: Neo4jGraph,
    num_results_to_display: int = 5
) -> Tuple[str, List[Dict]]:
    """
    Orchestrates Text-to-Cypher: generates query, executes, formats results.
    Returns an LLM-formatted explanation string and a list of movie data dictionaries.
    """
    logger.info(f"CYPHER_GEN: Answering graph query for: '{user_query}'")

    generated_cypher = generate_cypher_query_from_natural_language(user_query, openai_client)

    if not generated_cypher:
        return "I couldn't translate your question into a database query. Could you try rephrasing?", []

    cypher_results, error_message = execute_generated_cypher(generated_cypher, neo4j_graph_instance)

    if error_message:
        return error_message, [] # Return the user-friendly error message

    if not cypher_results:
        return f"I ran a query for '{user_query}' but didn't find any specific movies matching those exact criteria.", []

    # Format results for Streamlit display (MOVIE:/EXPLANATION:)
    llm_response_parts = []
    formatted_results_for_mapping = [] # This list will be used by app.py's map_llm_recs

    for i, movie_data_item in enumerate(cypher_results[:num_results_to_display]):
        title = movie_data_item.get('title', movie_data_item.get('m.title', f"Result {i+1}"))
        explanation_segments = []
        current_item_details = {'title': title} # For the list of dicts

        for key, value in movie_data_item.items():
            # Populate current_item_details for the list of dicts
            clean_key_map = key.replace('m.', '').replace('p.', '').replace('g.', '')
            if clean_key_map == 'tmdb_id': current_item_details['tmdb_id'] = value
            elif clean_key_map == 'overview': current_item_details['overview'] = value
            elif clean_key_map == 'tagline': current_item_details['tagline'] = value
            elif clean_key_map == 'poster_url': current_item_details['poster_url'] = value
            elif clean_key_map == 'trailer_url': current_item_details['trailer_url'] = value
            # ... add other properties you want to map for display ...

            # For the EXPLANATION string
            if key.lower() not in ['title', 'm.title', 'score'] and value is not None:
                clean_key_display = key.replace('m.', '').replace('p.', '').replace('g.', '').replace('_', ' ').capitalize()
                explanation_segments.append(f"{clean_key_display}: {value}")
        
        explanation_str = "; ".join(explanation_segments) if explanation_segments else "Details as per your query."
        current_item_details['explanation'] = explanation_str # Add to dict
        
        llm_response_parts.append(f"MOVIE: {title}\nEXPLANATION: {explanation_str}")
        formatted_results_for_mapping.append(current_item_details)

    llm_explanation_text = "\n\n".join(llm_response_parts)
    if not llm_explanation_text and cypher_results:
        llm_explanation_text = f"Found {len(cypher_results)} results. Displaying top {len(formatted_results_for_mapping)}."

    return llm_explanation_text, formatted_results_for_mapping

# --- Optional: main block for direct testing of this module ---
if __name__ == '__main__':
    print(f"Running {__file__} directly for testing.")
    # For direct testing, you'd need to initialize OpenAI and Neo4jGraph clients here
    # Example (requires config.py or environment variables for keys/URIs):
    # try:
    #     from config import settings
    #     test_openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    #     test_kg_instance = Neo4jGraph(
    #         url=settings.NEO4J_URI,
    #         username=settings.NEO4J_USERNAME,
    #         password=settings.NEO4J_PASSWORD,
    #         database=getattr(settings, 'NEO4J_DATABASE', "neo4j")
    #     )
    # except Exception as e:
    #     print(f"Test setup failed: {e}")
    #     test_openai_client = None
    #     test_kg_instance = None

    # if test_openai_client and test_kg_instance:
    #     test_query = "Movies starring Keanu Reeves"
    #     print(f"\nTesting Text-to-Cypher for: '{test_query}'")
    #     explanation, results = answer_graph_query_via_llm(test_query, test_openai_client, test_kg_instance, num_results_to_display=3)
    #     print("\nLLM Formatted Explanation:")
    #     print(explanation)
    #     print("\nStructured Results for Mapping:")
    #     for res in results:
    #         print(res)
    # else:
    #     print("Skipping direct test as OpenAI client or Neo4jGraph instance could not be initialized.")