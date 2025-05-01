import streamlit as st
import logging
import io
import pandas as pd
from PIL import Image

# --- Core Logic Imports ---
# Make sure these functions exist in your service/comparison modules
from core.recommendation_service import (
    process_input,
    get_comparison_results,
    initialize_services,
    shutdown_services,
    get_all_movie_titles_for_comparison, # Assuming you create this helper
    _driver as core_driver # Access driver if needed for helper
)
# Ensure the comparison function is imported if it's defined elsewhere
# from comparison.similarity_metrics import get_comparison_results # Example if separated

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="cineBoat 🎬 Multimodal RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed" # Optional: Collapse sidebar if you have one
)

# --- Initialize Backend Services ---
# Using session state to ensure initialization happens only once
if 'services_initialized' not in st.session_state:
    with st.spinner("Initializing backend services... Please wait."):
        try:
            log.info("Initializing services for Streamlit app...")
            initialize_services() # Call your initialization function
            st.session_state.services_initialized = True
            log.info("Services initialized successfully.")
            # Attempt to fetch movie titles once after initialization for the comparison dropdown
            try:
                 st.session_state.movie_titles_list = get_all_movie_titles_for_comparison(core_driver)
                 log.info(f"Fetched {len(st.session_state.movie_titles_list)} titles for comparison dropdown.")
            except Exception as title_e:
                 log.error(f"Failed to pre-fetch movie titles: {title_e}")
                 st.session_state.movie_titles_list = ["Error loading titles"]
        except Exception as e:
            log.error(f"FATAL: Failed to initialize core services: {e}", exc_info=True)
            st.error(f"Critical Error: Failed to initialize backend services: {e}. The app cannot function.")
            st.session_state.services_initialized = False
            # Stop the app if backend init fails critically
            st.stop()

# Check again if initialization failed after the attempt
if not st.session_state.get('services_initialized', False):
    st.error("Backend services failed to initialize. Please check logs or refresh.")
    st.stop()

# --- App Header ---
st.title("cineBoat 🎬: Multimodal Movie RAG")
st.caption("Ask questions about movies, identify posters, or compare recommendation methods!")

# --- Main Application Section ---
st.header("💬 Ask cineBoat")
st.markdown("Enter a text query below, or upload a movie poster image.")

# Use form to group inputs and have a single submit button
with st.form("rag_input_form"):
    col1, col2 = st.columns([3, 1]) # Text input wider
    with col1:
        user_query = st.text_input(
            "Your question:",
            placeholder="e.g., Who directed Inception? Tell me about this poster.",
            label_visibility="collapsed" # Hide label if context is clear
        )
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Poster Image",
            type=["jpg", "png", "jpeg"],
            label_visibility="collapsed" # Hide label
        )

    submitted = st.form_submit_button("Submit Query / Upload")

# --- Process RAG Input & Display Results ---
# Initialize session state for results if not present
if 'last_rag_response' not in st.session_state:
    st.session_state.last_rag_response = None
if 'uploaded_image_display' not in st.session_state:
    st.session_state.uploaded_image_display = None

image_bytes = None
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    # Store image bytes for display after form submission
    st.session_state.uploaded_image_display = image_bytes
else:
    # Clear previous image if none uploaded this time
    st.session_state.uploaded_image_display = None


if submitted and (user_query or image_bytes):
    with st.spinner('Thinking... 🤔'):
        try:
            # Call the core service with text and/or image bytes
            response = process_input(text_query=user_query, image_bytes=image_bytes)
            st.session_state.last_rag_response = response # Store result
        except Exception as e:
            log.error(f"Error processing input in Streamlit app: {e}", exc_info=True)
            st.error("An unexpected error occurred while processing your request.")
            st.session_state.last_rag_response = {"error": "Processing failed unexpectedly."} # Store error state
elif submitted:
    st.warning("Please enter a question or upload an image.")
    st.session_state.last_rag_response = None # Clear previous response

# --- Display RAG Results Area ---
st.markdown("---")
st.subheader("💡 Response")

rag_results_area = st.container() # Use container for cleaner updates

with rag_results_area:
    if st.session_state.uploaded_image_display:
        st.image(st.session_state.uploaded_image_display, caption="Uploaded Image Context", width=150)

    response = st.session_state.last_rag_response
    if response:
        if response.get("error"):
            st.error(f"Error: {response['error']}")
        else:
            if response.get("identified_movie"):
                st.info(f"ℹ️ Context based on identified movie: **{response['identified_movie']}**")
            elif response.get("image_analysis"): # If image uploaded but not identified
                 st.info(f"ℹ️ {response['image_analysis']}")

            st.markdown(response.get("answer", "Sorry, I couldn't formulate an answer."))
    else:
        st.markdown("_Ask a question or upload an image to get started!_")


# --- Comparison Section (Collapsible Expander) ---
st.markdown("---")
with st.expander("🔬 Compare Recommendation Methods"):
    st.markdown("See how different techniques find 'similar' movies based on plot keywords, plot meaning, or poster visuals.")
    st.caption("_This helps demonstrate the advantages of semantic and multimodal approaches._")

    movie_titles = st.session_state.get('movie_titles_list', ["Error loading titles"])

    if movie_titles and not movie_titles[0].startswith("Error"):
        selected_movie = st.selectbox(
            "Select a movie to compare:",
            movie_titles,
            key="comparison_movie_select" # Add unique key
            )

        if st.button("Compare Similarities", key="compare_button"):
            if selected_movie:
                with st.spinner(f"Calculating similarities for '{selected_movie}'..."):
                    # Call the comparison function from your service
                    comparison_results = get_comparison_results(selected_movie)
                    st.session_state.last_comparison_result = comparison_results # Store result
            else:
                st.warning("Please select a movie to compare.")
                st.session_state.last_comparison_result = None
    else:
         st.warning("Could not load movie list for comparison. Ensure data was loaded into the database.")
         st.session_state.last_comparison_result = None

    # --- Display Comparison Results ---
    comparison_results = st.session_state.get('last_comparison_result')
    if comparison_results:
        if comparison_results.get("error"):
            st.error(f"Comparison Error: {comparison_results['error']}")
        else:
            st.success(f"Comparison results for: **{comparison_results.get('target_movie')}**")

            # Display results side-by-side
            col_tfidf, col_text, col_image = st.columns(3)

            with col_tfidf:
                st.markdown("**Keyword (TF-IDF)**")
                st.caption("_Plot words frequency._")
                if comparison_results.get('tfidf'):
                    for item in comparison_results['tfidf']:
                        st.write(f"- {item['title']} ({item['score']:.3f})")
                else: st.warning("N/A")

            with col_text:
                st.markdown("**Semantic (Text Emb.)**")
                st.caption("_Plot meaning._")
                if comparison_results.get('text_embedding'):
                    for item in comparison_results['text_embedding']:
                        st.write(f"- {item['title']} ({item['score']:.3f})")
                else: st.warning("N/A")

            with col_image:
                st.markdown("**Visual (Poster Emb.)**")
                st.caption("_Poster style._")
                if comparison_results.get('image_embedding'):
                    for item in comparison_results['image_embedding']:
                        st.write(f"- {item['title']} ({item['score']:.3f})")
                else: st.warning("N/A")

            # Interpretation Text
            st.markdown("---")
            st.markdown("""
            **Interpretation:**
            *   **TF-IDF:** Often finds movies with similar specific words/names, but can miss thematic similarity.
            *   **Text Embedding:** Captures deeper meaning, themes, and style. Often feels more relevant.
            *   **Image Embedding:** Finds movies with visually similar poster styles (color, composition).
            *   **Graph RAG (Main App):** Combines semantic/visual understanding with graph relationships (actors, directors) and LLM reasoning for contextual answers/recommendations beyond simple similarity lists.
            """)

            if comparison_results.get('errors'):
                 st.warning(f"Note: Some comparison calculations failed: {'; '.join(comparison_results['errors'])}")

# --- Footer ---
st.markdown("---")
st.markdown("Powered by Neo4j, LangChain (Multimodal), CLIP, SentenceTransformers, Scikit-learn, and Streamlit.")
st.caption("cineBoat v1.0 - Multimodal RAG Demo")

# Cleanup function (Optional, see previous comments)
# import atexit
# atexit.register(shutdown_services)