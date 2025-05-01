import streamlit as st
import logging
from core.recommendation_service import get_recommendation_or_answer, initialize_services, shutdown_services

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="cineBoat 🎬",
    page_icon="🎬",
    layout="wide"
)

# --- Initialize Services ---
# Using a simple flag in session state to initialize only once per session
if 'services_initialized' not in st.session_state:
    try:
        log.info("Initializing services for Streamlit app...")
        initialize_services()
        st.session_state.services_initialized = True
        log.info("Services initialized successfully.")
    except Exception as e:
        log.error(f"FATAL: Failed to initialize core services: {e}")
        st.error(f"Failed to initialize backend services: {e}. Please check logs and configuration. The app may not function correctly.")
        # Optionally stop the app if services are critical: st.stop()
        st.session_state.services_initialized = False # Mark as failed

# --- App Header ---
st.title("cineBoat 🎬: Graph RAG Movie Recommendations")
st.caption("Ask for movie recommendations (e.g., 'movies like Inception') or ask questions about movies (e.g., 'Who directed Pulp Fiction?')")

# --- User Input ---
user_query = st.text_input("Enter your query:", placeholder="e.g., movies similar to The Matrix, who directed Interstellar?")

# --- Process Query and Display Results ---
if user_query and st.session_state.get('services_initialized', False):
    with st.spinner('Thinking... 🤔'):
        try:
            response = get_recommendation_or_answer(user_query)

            st.markdown("---") # Separator

            # --- Display based on response type ---
            response_type = response.get("type")

            if response_type == "recommendation":
                st.subheader(f"Recommendations similar to '{response.get('input_movie', 'your query')}':")
                st.caption(response.get("explanation", ""))
                recommendations = response.get("data", [])
                if recommendations:
                    # Display recommendations in columns
                    cols = st.columns(len(recommendations))
                    for i, movie in enumerate(recommendations):
                        with cols[i]:
                            if movie.get('poster_url'):
                                st.image(movie['poster_url'], caption=f"{movie.get('title', 'N/A')} ({movie.get('released', 'N/A')})", use_column_width=True)
                            else:
                                st.write(f"**{movie.get('title', 'N/A')}** ({movie.get('released', 'N/A')})")
                                st.caption("(No poster available)")
                            # Optional: Add similarity score
                            # score = movie.get('similarity_score')
                            # if score is not None:
                            #     st.caption(f"Similarity: {score:.2f}")
                else:
                    st.write("No recommendations found.")

            elif response_type == "answer":
                st.subheader("Answer:")
                st.markdown(response.get("data", "Sorry, I couldn't formulate an answer."))
                if response.get("context_used") is True:
                    st.caption("ℹ️ *Answer generated using context from the movie graph database.*")
                elif response.get("context_used") is False:
                     st.caption("ℹ️ *Answer generated based on general knowledge (no specific graph context found).*")


            elif response_type == "error":
                st.error(f"Error: {response.get('data', 'An unknown error occurred.')}")

            else:
                st.warning("Received an unknown response type.")

        except Exception as e:
            log.error(f"Error processing query in Streamlit app: {e}", exc_info=True)
            st.error("An unexpected error occurred while processing your request.")

elif user_query and not st.session_state.get('services_initialized', False):
     st.error("Backend services could not be initialized. Cannot process query.")


# --- Optional: Add Footer or About section ---
st.markdown("---")
st.markdown("Powered by Neo4j, LangChain, and Streamlit.")

# --- Cleanup Services on App Exit (Streamlit doesn't have a perfect hook) ---
# This is tricky with Streamlit's execution model. shutdown_services() might
# not always be called reliably. For simple cases, it might be okay, but
# resource management (like DB connections) often requires more robust handling
# in long-running deployments (e.g., separate process or connection pooling).
# Consider calling shutdown_services() if you add explicit exit buttons or similar.
# For this example, we rely on the Neo4j driver's built-in cleanup or OS handling.
# You *could* try registering an exit handler, but it's not guaranteed in Streamlit.
# import atexit
# atexit.register(shutdown_services) # Register cleanup function