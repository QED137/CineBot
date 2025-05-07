# import streamlit as st
# import logging
# import io
# import pandas as pd
# from PIL import Image

# # --- Core Logic Imports ---
# # Make sure these functions exist in your service/comparison modules
# from core.recommendation_service import (
#     process_input,
#     get_comparison_results,
#     initialize_services,
#     shutdown_services,
#     get_all_movie_titles_for_comparison, # Assuming you create this helper
#     _driver as core_driver # Access driver if needed for helper
# )
# # Ensure the comparison function is imported if it's defined elsewhere
# # from comparison.similarity_metrics import get_comparison_results # Example if separated

# # --- Logging Configuration ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="cineBoat 🎬 Multimodal RAG",
#     page_icon="🎬",
#     layout="wide",
#     initial_sidebar_state="collapsed" # Optional: Collapse sidebar if you have one
# )

# # --- Initialize Backend Services ---
# # Using session state to ensure initialization happens only once
# if 'services_initialized' not in st.session_state:
#     with st.spinner("Initializing backend services... Please wait."):
#         try:
#             log.info("Initializing services for Streamlit app...")
#             initialize_services() # Call your initialization function
#             st.session_state.services_initialized = True
#             log.info("Services initialized successfully.")
#             # Attempt to fetch movie titles once after initialization for the comparison dropdown
#             try:
#                  st.session_state.movie_titles_list = get_all_movie_titles_for_comparison(core_driver)
#                  log.info(f"Fetched {len(st.session_state.movie_titles_list)} titles for comparison dropdown.")
#             except Exception as title_e:
#                  log.error(f"Failed to pre-fetch movie titles: {title_e}")
#                  st.session_state.movie_titles_list = ["Error loading titles"]
#         except Exception as e:
#             log.error(f"FATAL: Failed to initialize core services: {e}", exc_info=True)
#             st.error(f"Critical Error: Failed to initialize backend services: {e}. The app cannot function.")
#             st.session_state.services_initialized = False
#             # Stop the app if backend init fails critically
#             st.stop()

# # Check again if initialization failed after the attempt
# if not st.session_state.get('services_initialized', False):
#     st.error("Backend services failed to initialize. Please check logs or refresh.")
#     st.stop()

# # --- App Header ---
# st.title("cineBoat 🎬: Multimodal Movie RAG")
# st.caption("Ask questions about movies, identify posters, or compare recommendation methods!")

# # --- Main Application Section ---
# st.header("💬 Ask cineBoat")
# st.markdown("Enter a text query below, or upload a movie poster image.")

# # Use form to group inputs and have a single submit button
# with st.form("rag_input_form"):
#     col1, col2 = st.columns([3, 1]) # Text input wider
#     with col1:
#         user_query = st.text_input(
#             "Your question:",
#             placeholder="e.g., Who directed Inception? Tell me about this poster.",
#             label_visibility="collapsed" # Hide label if context is clear
#         )
#     with col2:
#         uploaded_file = st.file_uploader(
#             "Upload Poster Image",
#             type=["jpg", "png", "jpeg"],
#             label_visibility="collapsed" # Hide label
#         )

#     submitted = st.form_submit_button("Submit Query / Upload")

# # --- Process RAG Input & Display Results ---
# # Initialize session state for results if not present
# if 'last_rag_response' not in st.session_state:
#     st.session_state.last_rag_response = None
# if 'uploaded_image_display' not in st.session_state:
#     st.session_state.uploaded_image_display = None

# image_bytes = None
# if uploaded_file is not None:
#     image_bytes = uploaded_file.getvalue()
#     # Store image bytes for display after form submission
#     st.session_state.uploaded_image_display = image_bytes
# else:
#     # Clear previous image if none uploaded this time
#     st.session_state.uploaded_image_display = None


# if submitted and (user_query or image_bytes):
#     with st.spinner('Thinking... 🤔'):
#         try:
#             # Call the core service with text and/or image bytes
#             response = process_input(text_query=user_query, image_bytes=image_bytes)
#             st.session_state.last_rag_response = response # Store result
#         except Exception as e:
#             log.error(f"Error processing input in Streamlit app: {e}", exc_info=True)
#             st.error("An unexpected error occurred while processing your request.")
#             st.session_state.last_rag_response = {"error": "Processing failed unexpectedly."} # Store error state
# elif submitted:
#     st.warning("Please enter a question or upload an image.")
#     st.session_state.last_rag_response = None # Clear previous response

# # --- Display RAG Results Area ---
# st.markdown("---")
# st.subheader("💡 Response")

# rag_results_area = st.container() # Use container for cleaner updates

# with rag_results_area:
#     if st.session_state.uploaded_image_display:
#         st.image(st.session_state.uploaded_image_display, caption="Uploaded Image Context", width=150)

#     response = st.session_state.last_rag_response
#     if response:
#         if response.get("error"):
#             st.error(f"Error: {response['error']}")
#         else:
#             if response.get("identified_movie"):
#                 st.info(f"ℹ️ Context based on identified movie: **{response['identified_movie']}**")
#             elif response.get("image_analysis"): # If image uploaded but not identified
#                  st.info(f"ℹ️ {response['image_analysis']}")

#             st.markdown(response.get("answer", "Sorry, I couldn't formulate an answer."))
#     else:
#         st.markdown("_Ask a question or upload an image to get started!_")


# # --- Comparison Section (Collapsible Expander) ---
# st.markdown("---")
# with st.expander("🔬 Compare Recommendation Methods"):
#     st.markdown("See how different techniques find 'similar' movies based on plot keywords, plot meaning, or poster visuals.")
#     st.caption("_This helps demonstrate the advantages of semantic and multimodal approaches._")

#     movie_titles = st.session_state.get('movie_titles_list', ["Error loading titles"])

#     if movie_titles and not movie_titles[0].startswith("Error"):
#         selected_movie = st.selectbox(
#             "Select a movie to compare:",
#             movie_titles,
#             key="comparison_movie_select" # Add unique key
#             )

#         if st.button("Compare Similarities", key="compare_button"):
#             if selected_movie:
#                 with st.spinner(f"Calculating similarities for '{selected_movie}'..."):
#                     # Call the comparison function from your service
#                     comparison_results = get_comparison_results(selected_movie)
#                     st.session_state.last_comparison_result = comparison_results # Store result
#             else:
#                 st.warning("Please select a movie to compare.")
#                 st.session_state.last_comparison_result = None
#     else:
#          st.warning("Could not load movie list for comparison. Ensure data was loaded into the database.")
#          st.session_state.last_comparison_result = None

#     # --- Display Comparison Results ---
#     comparison_results = st.session_state.get('last_comparison_result')
#     if comparison_results:
#         if comparison_results.get("error"):
#             st.error(f"Comparison Error: {comparison_results['error']}")
#         else:
#             st.success(f"Comparison results for: **{comparison_results.get('target_movie')}**")

#             # Display results side-by-side
#             col_tfidf, col_text, col_image = st.columns(3)

#             with col_tfidf:
#                 st.markdown("**Keyword (TF-IDF)**")
#                 st.caption("_Plot words frequency._")
#                 if comparison_results.get('tfidf'):
#                     for item in comparison_results['tfidf']:
#                         st.write(f"- {item['title']} ({item['score']:.3f})")
#                 else: st.warning("N/A")

#             with col_text:
#                 st.markdown("**Semantic (Text Emb.)**")
#                 st.caption("_Plot meaning._")
#                 if comparison_results.get('text_embedding'):
#                     for item in comparison_results['text_embedding']:
#                         st.write(f"- {item['title']} ({item['score']:.3f})")
#                 else: st.warning("N/A")

#             with col_image:
#                 st.markdown("**Visual (Poster Emb.)**")
#                 st.caption("_Poster style._")
#                 if comparison_results.get('image_embedding'):
#                     for item in comparison_results['image_embedding']:
#                         st.write(f"- {item['title']} ({item['score']:.3f})")
#                 else: st.warning("N/A")

#             # Interpretation Text
#             st.markdown("---")
#             st.markdown("""
#             **Interpretation:**
#             *   **TF-IDF:** Often finds movies with similar specific words/names, but can miss thematic similarity.
#             *   **Text Embedding:** Captures deeper meaning, themes, and style. Often feels more relevant.
#             *   **Image Embedding:** Finds movies with visually similar poster styles (color, composition).
#             *   **Graph RAG (Main App):** Combines semantic/visual understanding with graph relationships (actors, directors) and LLM reasoning for contextual answers/recommendations beyond simple similarity lists.
#             """)

#             if comparison_results.get('errors'):
#                  st.warning(f"Note: Some comparison calculations failed: {'; '.join(comparison_results['errors'])}")

# # --- Footer ---
# st.markdown("---")
# st.markdown("Powered by Neo4j, LangChain (Multimodal), CLIP, SentenceTransformers, Scikit-learn, and Streamlit.")
# st.caption("cineBoat v1.0 - Multimodal RAG Demo")

# # Cleanup function (Optional, see previous comments)
# # import atexit
# # atexit.register(shutdown_services)

###################################################################################################
# import streamlit as st
# from neo4j import GraphDatabase
# from PIL import Image
# import requests
# from io import BytesIO
# import numpy as np
# from transformers import CLIPProcessor, CLIPModel
# import torch
# from config import settings

# # Initialize Neo4j connection
# driver = GraphDatabase.driver(
#     settings.NEO4J_URI,
#     auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
# )

# # Load CLIP model for image-text similarity
# @st.cache_resource
# def load_clip_model():
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     return model, processor

# clip_model, clip_processor = load_clip_model()

# # Page config
# st.set_page_config(
#     page_title="CineMatch AI - Multimodal Movie Recommender",
#     page_icon="🎬",
#     layout="wide"
# )

# # CSS styling
# st.markdown("""
# <style>
#     .title { font-size: 2.5rem !important; color: #ff4b4b !important; }
#     .sidebar .sidebar-content { background-color: #0e1117; }
#     .movie-card { border-radius: 10px; padding: 15px; margin-bottom: 20px; 
#                   border-left: 5px solid #ff4b4b; background-color: #1a1d24; }
#     .movie-title { font-size: 1.4rem; color: #ffffff; }
#     .movie-tagline { font-style: italic; color: #aaaaaa; }
#     .similarity-badge { background-color: #ff4b4b; color: white; border-radius: 12px; 
#                        padding: 2px 10px; font-size: 0.8rem; }
# </style>
# """, unsafe_allow_html=True)

# # Helper functions
# def get_movie_by_id(tx, movie_id):
#     result = tx.run("""
#     MATCH (m:Movie {tmdb_id: $id})
#     OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
#     OPTIONAL MATCH (m)<-[r:ACTED_IN]-(a:Person)
#     RETURN m, COLLECT(DISTINCT d.name) AS directors, 
#            COLLECT(DISTINCT {name: a.name, role: r.roles[0]}) AS actors
#     """, id=movie_id)
#     return result.single()

# def semantic_search(tx, query_embedding, limit=10):
#     result = tx.run("""
#     WITH $embedding AS query_embedding
#     MATCH (m:Movie)
#     WHERE m.taglineEmbedding IS NOT NULL
#     RETURN m.tmdb_id AS id, m.title AS title, m.tagline AS tagline,
#            m.poster_url AS poster_url, m.overview AS overview,
#            gds.similarity.cosine(m.taglineEmbedding, query_embedding) AS similarity
#     ORDER BY similarity DESC
#     LIMIT $limit
#     """, embedding=query_embedding, limit=limit)
#     return list(result)

# def hybrid_search(tx, text_query=None, image_embedding=None, limit=10):
#     query = """
#     CALL {
#         // Text-based semantic search
#         WITH $text_embedding AS text_embedding
#         MATCH (m:Movie)
#         WHERE m.taglineEmbedding IS NOT NULL AND $text_query IS NOT NULL
#         WITH m, gds.similarity.cosine(m.taglineEmbedding, text_embedding) AS text_score
#         ORDER BY text_score DESC
#         LIMIT 50
#         RETURN m, text_score
        
#         UNION
        
#         // Image-based visual search
#         WITH $image_embedding AS image_embedding
#         MATCH (m:Movie)
#         WHERE m.poster_embedding IS NOT NULL AND $image_embedding IS NOT NULL
#         WITH m, gds.similarity.cosine(m.poster_embedding, image_embedding) AS image_score
#         ORDER BY image_score DESC
#         LIMIT 50
#         RETURN m, image_score
#     }
#     WITH m, 
#          COALESCE(text_score, 0) * COALESCE($text_weight, 0) + 
#          COALESCE(image_score, 0) * COALESCE($image_weight, 1) AS combined_score
#     RETURN m.tmdb_id AS id, m.title AS title, m.tagline AS tagline,
#            m.poster_url AS poster_url, m.overview AS overview,
#            combined_score AS similarity
#     ORDER BY combined_score DESC
#     LIMIT $limit
#     """
#     params = {
#         "text_query": text_query,
#         "text_embedding": get_text_embedding(text_query) if text_query else None,
#         "image_embedding": image_embedding,
#         "text_weight": 0.7 if text_query else 0,
#         "image_weight": 0.3 if image_embedding else 0,
#         "limit": limit
#     }
#     result = tx.run(query, params)
#     return list(result)

# def get_text_embedding(text):
#     # Use your existing embedding generation method
#     with driver.session() as session:
#         result = session.run("""
#         RETURN genai.vector.encode(
#             $text,
#             "OpenAI",
#             {
#                 token: $apiKey,
#                 endpoint: $endpoint
#             }
#         ) AS embedding
#         """, text=text, apiKey=settings.OPENAI_API_KEY, endpoint=settings.OPENAI_ENDPOINT)
#         return result.single()["embedding"]

# def get_image_embedding(image):
#     inputs = clip_processor(images=image, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         image_features = clip_model.get_image_features(**inputs)
#     return image_features[0].numpy().tolist()

# def display_movie(movie, similarity=None):
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         if movie["poster_url"]:
#             try:
#                 response = requests.get(movie["poster_url"])
#                 img = Image.open(BytesIO(response.content))
#                 st.image(img, width=150)
#             except:
#                 st.image("https://via.placeholder.com/150x225?text=No+Poster", width=150)
    
#     with col2:
#         st.markdown(f'<p class="movie-title">{movie["title"]}</p>', unsafe_allow_html=True)
#         if movie["tagline"]:
#             st.markdown(f'<p class="movie-tagline">"{movie["tagline"]}"</p>', unsafe_allow_html=True)
#         if similarity:
#             st.markdown(f'<span class="similarity-badge">{similarity:.0%} match</span>', 
#                         unsafe_allow_html=True)
#         if movie["overview"]:
#             st.caption(movie["overview"])
#         st.markdown("---")

# # Main app
# def main():
#     st.markdown('<h1 class="title">CineMatch AI</h1>', unsafe_allow_html=True)
#     st.markdown("### Your Intelligent Multimodal Movie Recommender")
    
#     # Sidebar controls
#     with st.sidebar:
#         st.header("Search Options")
#         search_mode = st.radio(
#             "Search by:",
#             ["Text Description", "Image", "Hybrid (Text + Image)"],
#             index=0
#         )
        
#         if search_mode in ["Text Description", "Hybrid (Text + Image)"]:
#             text_query = st.text_area(
#                 "Describe what you're looking for:",
#                 "A sci-fi movie about artificial intelligence with philosophical themes"
#             )
#         else:
#             text_query = None
            
#         if search_mode in ["Image", "Hybrid (Text + Image)"]:
#             uploaded_file = st.file_uploader(
#                 "Upload a movie poster or similar image:",
#                 type=["jpg", "jpeg", "png"]
#             )
#             if uploaded_file:
#                 image = Image.open(uploaded_file)
#                 st.image(image, caption="Uploaded Image", use_column_width=True)
#             else:
#                 image = None
#         else:
#             image = None
            
#         num_results = st.slider("Number of recommendations:", 5, 20, 10)
        
#         if st.button("Get Recommendations"):
#             with st.spinner("Finding your perfect movies..."):
#                 try:
#                     with driver.session() as session:
#                         if search_mode == "Text Description":
#                             embedding = get_text_embedding(text_query)
#                             results = session.read_transaction(
#                                 semantic_search, 
#                                 embedding, 
#                                 num_results
#                             )
#                         elif search_mode == "Image":
#                             embedding = get_image_embedding(image)
#                             results = session.read_transaction(
#                                 hybrid_search, 
#                                 None,
#                                 embedding,
#                                 num_results
#                             )
#                         else:  # Hybrid
#                             text_embedding = get_text_embedding(text_query) if text_query else None
#                             image_embedding = get_image_embedding(image) if image else None
#                             results = session.read_transaction(
#                                 hybrid_search, 
#                                 text_query,
#                                 image_embedding,
#                                 num_results
#                             )
                            
#                     st.session_state.results = results
                    
#                 except Exception as e:
#                     st.error(f"Error generating recommendations: {str(e)}")
    
#     # Main results display
#     if "results" in st.session_state and st.session_state.results:
#         st.header("Recommended Movies")
#         for movie in st.session_state.results:
#             with st.container():
#                 display_movie(movie, movie["similarity"])
#     else:
#         st.info("""
#         🎬 Welcome to CineMatch AI!  
#         Use the sidebar to:
#         - Search by text description (e.g., "funny superhero movies")
#         - Upload an image (poster or scene)
#         - Combine both for hybrid recommendations
#         """)
#         st.image("https://via.placeholder.com/800x400?text=CineMatch+AI+Movie+Recommender", 
#                 use_column_width=True)

# if __name__ == "__main__":
#     main()

# app.py
import streamlit as st
from PIL import Image
import io # For handling image bytes
import re # For parsing LLM output
from typing import List, Dict, Optional # For type hinting

# Import your RAG functions and logger from rag_core.py
# Make sure core/core_rag.py has the RAG functions that return a tuple:
# (llm_explanation_text, initial_retrieved_movies)
from core.core_rag import recommend_by_text, recommend_by_poster_image, logger

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

# --- Page Configuration ---
st.set_page_config(
    page_title="CineBot - Multimodal Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- App Header ---
# st.image("https://www.themoviedb.org/t/p/original/wwemzKWzjKYJFfCeiB57q3r4Bcm.svg", width=200) # Optional logo
st.title("🎬 CineBot - Your Intelligent Movie Recommender")
st.caption("Discover movies based on text descriptions or poster images!")

# --- Session State (to keep track of structured recommendations) ---
if 'text_recommendations_detailed' not in st.session_state:
    st.session_state.text_recommendations_detailed = [] # Will store list of dicts
if 'image_recommendations_detailed' not in st.session_state:
    st.session_state.image_recommendations_detailed = [] # Will store list of dicts
if 'last_text_query' not in st.session_state:
    st.session_state.last_text_query = ""
if 'last_image_filename' not in st.session_state:
    st.session_state.last_image_filename = ""

# --- Helper function to parse LLM output ---
def parse_llm_recommendations(llm_text_response: str) -> List[Dict]:
    recommendations = []
    pattern = re.compile(r"MOVIE:\s*(.*?)\s*\n\s*EXPLANATION:\s*(.*?)(?=\n\nMOVIE:|\Z)", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(llm_text_response)

    for match in matches:
        title = match[0].strip()
        explanation = match[1].strip()
        recommendations.append({"title": title, "explanation": explanation})
    
    if not recommendations and llm_text_response and "I'm sorry" not in llm_text_response and "I couldn't find" not in llm_text_response:
        logger.warning("LLM response was not in the expected structured format. Displaying as raw text.")
        return [{"title": "CineBot's Thoughts", "explanation": llm_text_response}]
    return recommendations

# --- Helper to map LLM recommended titles back to fully detailed retrieved movies ---
def map_llm_recs_to_retrieved_details(
    llm_parsed_recommendations: List[Dict],
    initially_retrieved_movies: List[Dict]
) -> List[Dict]:
    detailed_recommendations = []
    if not initially_retrieved_movies: # Handle case where retrieval failed
        logger.warning("No initially retrieved movies to map LLM recommendations to.")
        # Return LLM parsed recs as is, poster/trailer will be missing
        return [{"title": rec.get("title"), "explanation": rec.get("explanation")} for rec in llm_parsed_recommendations]

    retrieved_lookup = {movie.get('title','').lower().strip(): movie for movie in initially_retrieved_movies}

    for llm_rec in llm_parsed_recommendations:
        llm_title_lower = llm_rec.get('title','').lower().strip()
        matched_movie_data = retrieved_lookup.get(llm_title_lower)

        if matched_movie_data:
            detailed_rec = {
                "title": matched_movie_data.get('title'),
                "explanation": llm_rec.get('explanation'),
                "poster_url": matched_movie_data.get('poster_url'),
                "trailer_url": matched_movie_data.get('trailer_url'),
                "tagline": matched_movie_data.get('tagline'),
                "overview": matched_movie_data.get('overview'),
                "tmdb_id": matched_movie_data.get('tmdb_id')
            }
            detailed_recommendations.append(detailed_rec)
        else:
            logger.warning(f"Could not map LLM recommended title '{llm_rec.get('title')}' back to full details. Check LLM prompt to return EXACT titles from context.")
            detailed_recommendations.append({
                "title": llm_rec.get("title"),
                "explanation": llm_rec.get("explanation"),
                "poster_url": None,
                "trailer_url": None
            })
    return detailed_recommendations

# --- Function to display recommendations as cards ---
def display_recommendation_cards_v2(detailed_recommendations: List[Dict]):
    if not detailed_recommendations:
        return # Caller should handle "no recommendations" message

    cols_per_row = st.session_state.get('cols_per_row_slider', 2) # Get from session state

    for i in range(0, len(detailed_recommendations), cols_per_row):
        cols = st.columns(cols_per_row)
        batch_recs = detailed_recommendations[i:i+cols_per_row]

        for idx, rec in enumerate(batch_recs):
            if idx < len(cols):
                with cols[idx]:
                    # Use a unique key part for each container based on movie ID or index
                    unique_key_part = rec.get('tmdb_id', f"rec_{i}_{idx}")
                    container = st.container(border=True, key=f"card_container_{unique_key_part}")
                    
                    container.subheader(rec.get("title", "Recommendation"))

                    poster_url = rec.get("poster_url")
                    if poster_url and poster_url.startswith("http"):
                        container.image(poster_url, use_container_width=True)
                    else:
                        container.caption(f"Poster for {rec.get('title', '')} (Not available)")
                    
                    if rec.get("explanation"):
                        container.markdown(f"**CineBot says:** {rec.get('explanation')}")
                    else:
                        container.markdown("_CineBot is speechless about this one!_")

                    trailer_url = rec.get("trailer_url")
                    # Use expander directly, button approach is complex with Streamlit reruns
                    if trailer_url and "youtube.com/watch?v=" in trailer_url:
                         with container.expander(f"🎬 Trailer for {rec.get('title', '')}", expanded=False):
                            st.video(trailer_url)
                    elif trailer_url:
                        container.markdown(f"[🎬 Watch Trailer Link]({trailer_url})")
                    
                    # More details expander
                    has_more_details = rec.get("tagline") or rec.get("overview")
                    if has_more_details:
                        with container.expander(f"More Details", expanded=False):
                            if rec.get("tagline"):
                                st.markdown(f"**Tagline:** {rec.get('tagline')}")
                            if rec.get("overview"):
                                st.markdown(f"**Overview:** {rec.get('overview')}")
                    # Add a little space
                    container.markdown("<br>", unsafe_allow_html=True)


# --- UI Elements ---
st.sidebar.header("Display Options")
st.session_state.cols_per_row_slider = st.sidebar.slider("Movies per row:", 1, 4, 2, key="cols_slider")


tab1, tab2 = st.tabs(["🔍 Recommend by Text", "🖼️ Recommend by Poster Image"])

with tab1:
    st.header("Describe Your Desired Movie")
    text_query = st.text_area(
        "E.g., 'A heartwarming animated film about friendship and adventure, perfect for families.'",
        key="text_query_input_area", # Changed key to avoid conflict if you had text_input before
        value=st.session_state.last_text_query,
        height=100
    )

    if st.button("Get Text-Based Recommendations", key="text_submit_btn", type="primary"):
        if text_query:
            st.session_state.last_text_query = text_query
            st.session_state.text_recommendations_detailed = [] # Clear previous before new search
            st.session_state.image_recommendations_detailed = []
            with st.spinner("CineBot is thinking... 🧠 (This might take a moment)"):
                try:
                    # Assumes recommend_by_text returns (llm_response_text, initial_retrieved_movies)
                    llm_response_text, initial_retrieved_movies = recommend_by_text(
                        text_query, top_k_retrieval=5, num_recommendations=3
                    )
                    parsed_llm_recs = parse_llm_recommendations(llm_response_text)
                    detailed_recs = map_llm_recs_to_retrieved_details(parsed_llm_recs, initial_retrieved_movies)
                    st.session_state.text_recommendations_detailed = detailed_recs
                except Exception as e:
                    logger.error(f"Error in text recommendation flow: {e}", exc_info=True)
                    st.error("Oops! Something went wrong. CineBot is a bit confused right now.")
        else:
            st.warning("Please enter a description to get recommendations.")

    if st.session_state.text_recommendations_detailed:
        st.markdown("---")
        st.header("CineBot's Picks For You (based on text):")
        display_recommendation_cards_v2(st.session_state.text_recommendations_detailed)
    elif st.session_state.last_text_query and not st.session_state.text_recommendations_detailed and st.button("Retry Text Search?", key="retry_text_search"):
        # This button is a bit simplistic, just for an idea
        pass # Rerun logic would be complex here, simpler to let user click original button


with tab2:
    st.header("Find Movies by Poster Likeness")
    uploaded_image = st.file_uploader(
        "Upload a movie poster image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader_widget" # Changed key
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption=f"Your query poster: {uploaded_image.name}", width=200)

        if st.button("Get Image-Based Recommendations", key="image_submit_btn", type="primary"):
            st.session_state.last_image_filename = uploaded_image.name
            st.session_state.text_recommendations_detailed = []
            st.session_state.image_recommendations_detailed = [] # Clear previous
            with st.spinner("CineBot is analyzing the poster... 🎨 (This might take a moment)"):
                try:
                    image_bytes = uploaded_image.getvalue()
                    # Assumes recommend_by_poster_image returns (llm_response_text, initial_retrieved_movies)
                    llm_response_text, initial_retrieved_movies = recommend_by_poster_image(
                        image_bytes, top_k_retrieval=5, num_recommendations=3
                    )
                    parsed_llm_recs = parse_llm_recommendations(llm_response_text)
                    detailed_recs = map_llm_recs_to_retrieved_details(parsed_llm_recs, initial_retrieved_movies)
                    st.session_state.image_recommendations_detailed = detailed_recs
                except Exception as e:
                    logger.error(f"Error in image recommendation flow: {e}", exc_info=True)
                    st.error("Oops! Something went wrong. CineBot needs its glasses.")

    if st.session_state.image_recommendations_detailed:
        st.markdown("---")
        st.header("CineBot's Picks For You (based on image):")
        display_recommendation_cards_v2(st.session_state.image_recommendations_detailed)


# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Powered by Neo4j | OpenAI | CLIP | Streamlit</p>", unsafe_allow_html=True)

logger.info("Streamlit app initialized/reloaded.")