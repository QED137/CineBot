# core/cypher_query_generation.py

import logging
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from langchain_neo4j import Neo4jGraph # If executing Cypher here

# --- Logger Setup ---
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Schema and Few-Shot Examples ---
NEO4J_MOVIE_SCHEMA_FOR_LLM = """
The Neo4j graph contains information about movies.
Node labels and their relevant properties:
- Movie:
  - tmdb_id: STRING (The Movie Database ID, unique identifier)
  - title: STRING (The title of the movie)
  - overview: STRING (A brief summary of the movie)
  - tagline: STRING (The movie's tagline)
  - release_date: DATE (A native Neo4j Date object, e.g., date({year:2010, month:7, day:16}) )
    - Access year using `m.release_date.year`.
    - Access month using `m.release_date.month`.
    - Access day using `m.release_date.day`.
  - vote_average: FLOAT (Average user rating)
  - poster_url: STRING (URL to the movie poster)
  - trailer_url: STRING (URL to the movie trailer)
  # Add other relevant properties like runtime, budget, revenue, popularity
- Person:
  - tmdb_id: STRING
  - name: STRING (Name of the person)
  # Add other relevant Person properties like biography, birthday
- Genre:
  - name: STRING (Name of the genre, e.g., "Action", "Sci-Fi")

Relationships:
- (:Person)-[:DIRECTED]->(:Movie)
- (:Person)-[:ACTED_IN {roles: LIST<STRING>}]->(:Movie) (Relationship can have 'roles' property)
- (:Movie)-[:HAS_GENRE]->(:Genre) # <<< IMPORTANT: VERIFY THIS RELATIONSHIP TYPE AND DIRECTION EXACTLY MATCHES YOUR GRAPH

Instructions for Cypher Generation:
- To filter by year, use the '.year' component of the 'release_date' DATE property (e.g., `m.release_date.year > 2005`).
- For names (people, genres), use exact matching unless the query implies partial matching (e.g., "movies with 'Matrix' in title" -> `m.title CONTAINS 'Matrix'`).
- Return useful movie properties: tmdb_id, title, overview, release_date, poster_url.
- Limit results to a reasonable number (e.g., LIMIT 15) if not specified by the user.
- If a query is too vague or cannot be translated, return "QUERY_NOT_FEASIBLE".
- Only output the Cypher query itself, no explanations or introductory text.
"""

FEW_SHOT_EXAMPLES_FOR_LLM = """
User Query: "Movies directed by Christopher Nolan"
Cypher Query: MATCH (p:Person {name: 'Christopher Nolan'})-[:DIRECTED]->(m:Movie) RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview ORDER BY m.release_date DESC LIMIT 15

User Query: "What movies did Tom Hanks star in after 2010?"
Cypher Query: MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) WHERE m.release_date.year > 2010 RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview ORDER BY m.release_date DESC LIMIT 15

User Query: "Sci-Fi movies released in the year 2023"
Cypher Query: MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre {name: 'Science Fiction'}) WHERE m.release_date.year = 2023 RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview LIMIT 15

User Query: "Movies released before 1995"
Cypher Query: MATCH (m:Movie) WHERE m.release_date.year < 1995 RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview ORDER BY m.release_date DESC LIMIT 15

User Query: "Movies after 2005"
Cypher Query: MATCH (m:Movie) WHERE m.release_date.year > 2005 RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview ORDER BY m.release_date.year ASC LIMIT 15

User Query: "Movies released between 2000 and 2005"
Cypher Query: MATCH (m:Movie) WHERE m.release_date.year >= 2000 AND m.release_date.year <= 2005 RETURN m.title AS title, m.release_date AS release_date, m.overview AS overview ORDER BY m.release_date.year ASC LIMIT 15

User Query: "Movies with high ratings and budget over 100 million"
Cypher Query: MATCH (m:Movie) WHERE m.vote_average > 8.0 AND m.budget > 100000000 RETURN m.title, m.vote_average, m.budget, m.overview ORDER BY m.vote_average DESC LIMIT 15
"""

# --- Core Text-to-Cypher Generation Function ---
def generate_cypher_query_from_natural_language(
    user_query: str,
    openai_client: OpenAI,
    llm_model: str = "gpt-3.5-turbo-0125"
) -> Optional[str]:
    # ... (This function's implementation remains the same) ...
    if not openai_client:
        logger.error("OpenAI client not provided to generate_cypher_query.")
        return None
    if not user_query.strip():
        logger.warning("Empty user query provided for Cypher generation.")
        return None

    prompt = f"""You are an expert Neo4j Cypher query generator.
Your task is to convert the user's natural language question about movies into a Cypher query.
You must use the provided schema and follow the instructions precisely.

Schema:
{NEO4J_MOVIE_SCHEMA_FOR_LLM}

Examples of User Queries and desired Cypher Output:
{FEW_SHOT_EXAMPLES_FOR_LLM}

Now, generate a Cypher query for the following user request:
User Query: "{user_query}"
Cypher Query:"""
    # ... (rest of the function as before) ...
    try:
        logger.info(f"Sending prompt to LLM for Cypher generation (query: '{user_query[:70]}...')")
        completion = openai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=350
        )
        generated_cypher = completion.choices[0].message.content.strip()
        if generated_cypher.startswith("```cypher"):
            generated_cypher = generated_cypher[len("```cypher"):].strip()
        if generated_cypher.startswith("```"):
            generated_cypher = generated_cypher[len("```"):].strip()
        if generated_cypher.endswith("```"):
            generated_cypher = generated_cypher[:-len("```")].strip()

        logger.info(f"LLM generated Cypher: {generated_cypher}")

        if "QUERY_NOT_FEASIBLE" in generated_cypher.upper() or \
           not generated_cypher.strip().upper().startswith("MATCH"):
            logger.warning(f"LLM indicated query is not feasible or generated invalid Cypher start: {generated_cypher}")
            return None
        return generated_cypher
    except Exception as e:
        logger.error(f"Error generating Cypher from LLM: {e}", exc_info=True)
        return None


# --- Function to Execute Cypher ---
def execute_generated_cypher(
    cypher_query: str,
    neo4j_graph_instance: Neo4jGraph
) -> Tuple[Optional[List[Dict]], Optional[str]]:
    # ... (This function remains the same) ...
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
        # Check for common GqlError messages or Neo4j driver errors
        elif "resolve" in err_str and ("variable" in err_str or "property" in err_str or "label" in err_str or "type" in err_str):
             return None, "I couldn't find some specific information (like a movie property, person, genre, or relationship type) mentioned in your query. Please check the spelling or try rephrasing."
        else: return None, "There was an issue querying the movie database. Please try again."


# --- Top-Level Orchestrator Function ---
def answer_graph_query_via_llm(
    user_query: str,
    openai_client: OpenAI,
    neo4j_graph_instance: Neo4jGraph,
    num_results_to_display: int = 5
) -> Tuple[str, List[Dict]]:
    # ... (This function's formatting logic for dates needs to be aware that release_date is a Neo4j Date object) ...
    logger.info(f"CYPHER_GEN: Answering graph query for: '{user_query}'")
    generated_cypher = generate_cypher_query_from_natural_language(user_query, openai_client)

    if not generated_cypher:
        return "I couldn't translate your question into a database query. Could you try rephrasing, perhaps asking about movie properties like director, actor, genre, or release year more explicitly?", []

    cypher_results, error_message = execute_generated_cypher(generated_cypher, neo4j_graph_instance)

    if error_message:
        return f"When trying to answer '{user_query}': {error_message}", []

    if not cypher_results:
        return f"I ran a query for '{user_query}' but didn't find any specific movies matching those exact criteria in the database.", []

    llm_response_parts = []
    formatted_results_for_mapping = []
    for i, movie_data_item in enumerate(cypher_results[:num_results_to_display]):
        title = movie_data_item.get('title', movie_data_item.get('m.title', f"Result {i+1} (Title N/A)"))
        explanation_segments = []
        current_item_details = {'title': title}

        for key, value in movie_data_item.items():
            clean_key_map = key.replace('m.', '').replace('p.', '').replace('g.', '')
            
            # Handle Neo4j Date/DateTime objects for display and mapping
            display_value = value
            if hasattr(value, 'isoformat'): # Check for neo4j.time.Date or similar
                display_value = value.isoformat()

            # Populate current_item_details
            if clean_key_map == 'tmdb_id': current_item_details['tmdb_id'] = display_value
            elif clean_key_map == 'overview': current_item_details['overview'] = display_value
            elif clean_key_map == 'tagline': current_item_details['tagline'] = display_value
            elif clean_key_map == 'poster_url': current_item_details['poster_url'] = display_value
            elif clean_key_map == 'trailer_url': current_item_details['trailer_url'] = display_value
            elif clean_key_map == 'release_date': current_item_details['release_date'] = display_value # Already a string now
            # Add other properties explicitly if needed for cards

            # For the EXPLANATION string
            if key.lower() not in ['title', 'm.title', 'score', 'tmdb_id', 'poster_url', 'trailer_url', 'overview', 'tagline'] and value is not None:
                clean_key_display = clean_key_map.replace('_', ' ').capitalize()
                explanation_segments.append(f"{clean_key_display}: {display_value}")
        
        explanation_str = "; ".join(explanation_segments) if explanation_segments else "Details as per your query."
        overview_snippet = current_item_details.get('overview', '')
        if overview_snippet and len(explanation_segments) < 2 :
             explanation_str += f" Overview: {str(overview_snippet)[:100]}..." # Ensure overview is str
        
        current_item_details['explanation'] = explanation_str.strip()
        llm_response_parts.append(f"MOVIE: {title}\nEXPLANATION: {current_item_details['explanation']}")
        formatted_results_for_mapping.append(current_item_details)

    llm_explanation_text = "\n\n".join(llm_response_parts)
    if not llm_explanation_text and cypher_results:
        llm_explanation_text = f"Found {len(cypher_results)} results matching your query '{user_query}'. Displaying up to {len(formatted_results_for_mapping)}."
    return llm_explanation_text, formatted_results_for_mapping


# --- Optional: main block for direct testing of this module ---
if __name__ == '__main__':
    # ... (your __main__ block for testing as before) ...
    print(f"Running {__file__} directly for testing.")
    try:
        from config import settings
        test_openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        test_kg_instance = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=getattr(settings, 'NEO4J_DATABASE', "neo4j")
        )
        CLIENTS_INITIALIZED = True
    except Exception as e:
        print(f"Test setup failed (OpenAI/Neo4j client init): {e}")
        CLIENTS_INITIALIZED = False

    if CLIENTS_INITIALIZED:
        test_queries = [
            "Movies directed by Christopher Nolan", # Should work as before
            "Movies after 2005", # Should now use m.release_date.year
            "What are some action movies from 2022?", # Will test HAS_GENRE and m.release_date.year
            "Tom Hanks films released before the year 2000" # Should now use m.release_date.year
        ]
        for tq in test_queries:
            print(f"\n--- Testing User Query: '{tq}' ---")
            explanation, results = answer_graph_query_via_llm(
                tq,
                test_openai_client,
                test_kg_instance,
                num_results_to_display=3
            )
            print("\nLLM Formatted Explanation String:")
            print(explanation)
            print("\nStructured Results for Mapping/Display:")
            if results:
                for i, res in enumerate(results):
                    print(f"  Result {i+1}: {res}")
            else:
                print("  No structured results returned.")
    else:
        print("Skipping direct test as OpenAI client or Neo4jGraph instance could not be initialized.")