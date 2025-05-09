# app.py
import streamlit as st
from PIL import Image
import io # For handling image bytes
import re # For parsing LLM output
from typing import List, Dict, Optional # For type hinting

# Import your RAG functions and logger from rag_core.py
# Make sure core/core_rag.py has the RAG functions that return a tuple:
# (llm_explanation_text, initial_retrieved_movies)
# Mocking the import if core_rag is not available for testing
try:
    from core.core_rag import recommend_by_text, recommend_by_poster_image, logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    def mock_recommendation(query_or_bytes, top_k_retrieval=3, num_recommendations=2):
        logger.info(f"Mock recommendation called with: {type(query_or_bytes)}")
        mock_movies = [
            {"title": f"Mock Movie {i+1}", "explanation": "This is a great mock movie because reasons.",
             "poster_url": f"https://picsum.photos/seed/{i+1}/200/300",
             "trailer_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", # Placeholder
             "tagline": f"An amazing tagline for movie {i+1}",
             "overview": f"A detailed overview of the spectacular mock movie {i+1}. It involves adventure and excitement.",
             "tmdb_id": f"mock_tmdb_{i+1}"}
            for i in range(num_recommendations)
        ]
        llm_response = "\n\n".join([f"MOVIE: {m['title']}\nEXPLANATION: {m['explanation']}" for m in mock_movies])
        return llm_response, mock_movies # Return LLM text and detailed list

    recommend_by_text = mock_recommendation
    recommend_by_poster_image = mock_recommendation
    logger.warning("Using MOCK RAG functions as core.core_rag was not found.")


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage - Keep if needed

# --- Page Configuration ---
st.set_page_config(
    page_title="CineBot - Multimodal Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- App Header ---
# Note box at the top
st.markdown(
    """
    <style>
    .fancy-marquee-container {
        height: 42px;
        overflow: hidden;
        position: relative;
        background: rgba(0, 172, 193, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(0, 172, 193, 0.2);
        box-shadow: 0 0 8px rgba(0, 172, 193, 0.2);
        backdrop-filter: blur(6px);
        padding-left: 10px;
    }

    .fancy-marquee-text {
        position: absolute;
        width: 100%;
        height: 100%;
        margin: 0;
        line-height: 42px;
        font-size: 15px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 500;
        background: linear-gradient(90deg, #00e5ff, #18ffff, #00acc1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: #00e5ff; /* fallback color */
        text-shadow: 0 0 6px rgba(0, 172, 193, 0.3);
        animation: scrollLeft 15s linear infinite;
    }

    @keyframes scrollLeft {
        0%   { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    </style>

    <div class="fancy-marquee-container">
        <p class="fancy-marquee-text">
        ⚠️ Demo runs on Neo4j Free Tier — limited graph scale and no vector indexing may affect RAG performance. Not for production use.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



st.title("🎬 CineBot - Intelligent Movie Recommender") # Slightly shorter title
st.caption("Discover movies using text descriptions or poster images!")

# --- Session State ---
if 'text_recommendations_detailed' not in st.session_state:
    st.session_state.text_recommendations_detailed = []
if 'image_recommendations_detailed' not in st.session_state:
    st.session_state.image_recommendations_detailed = []
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
    if not initially_retrieved_movies:
        logger.warning("No initially retrieved movies to map LLM recommendations to.")
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
            logger.warning(f"Could not map LLM recommended title '{llm_rec.get('title')}' back to full details.")
            detailed_recommendations.append({
                "title": llm_rec.get("title"),
                "explanation": llm_rec.get("explanation"),
                "poster_url": None,
                "trailer_url": None
            })
    return detailed_recommendations

# --- Function to display recommendations as cards (MODIFIED FOR COMPACTNESS) ---
def display_recommendation_cards_v2(detailed_recommendations: List[Dict]):
    if not detailed_recommendations:
        return

    cols_per_row = st.session_state.get('cols_per_row_slider', 3) # Default to 3

    for i in range(0, len(detailed_recommendations), cols_per_row):
        cols = st.columns(cols_per_row)
        batch_recs = detailed_recommendations[i:i+cols_per_row]

        for idx, rec in enumerate(batch_recs):
            if idx < len(cols):
                with cols[idx]:
                    unique_key_part = rec.get('tmdb_id', f"rec_{i}_{idx}")
                    with st.container(border=True, key=f"card_container_{unique_key_part}"):
                        st.markdown(f"**{rec.get('title', 'Recommendation')}**") # More compact title

                        poster_url = rec.get("poster_url")
                        if poster_url and poster_url.startswith("http"):
                            st.image(poster_url, use_column_width=True) # Scales with column
                        else:
                            st.caption(f"Poster for {rec.get('title', '')} (N/A)")

                        if rec.get("explanation"):
                            # Smaller text for explanation
                            st.markdown(f"<small><i>CineBot says:</i> {rec.get('explanation')}</small>", unsafe_allow_html=True)
                        else:
                            st.caption("_CineBot is speechless!_")

                        trailer_url = rec.get("trailer_url")
                        if trailer_url and "youtube.com/watch?v=" in trailer_url:
                             with st.expander("Trailer", expanded=False): # Shorter label
                                st.video(trailer_url)
                        elif trailer_url:
                            st.markdown(f"<small>[🎬 Watch Trailer]({trailer_url})</small>", unsafe_allow_html=True) # Smaller link

                        has_more_details = rec.get("tagline") or rec.get("overview")
                        if has_more_details:
                            with st.expander("Details", expanded=False): # Shorter label
                                if rec.get("tagline"):
                                    st.caption(f"Tagline: {rec.get('tagline')}") # Caption for less emphasis
                                if rec.get("overview"):
                                    st.caption(f"Overview: {rec.get('overview')}")
                        # No explicit <br> to save vertical space

# --- UI Elements ---
st.sidebar.header("Display Options")
st.session_state.cols_per_row_slider = st.sidebar.slider(
    "Movies per row:",
    min_value=1,
    max_value=4,  # Max 4 cards per row seems reasonable for most screens
    value=3,      # Default to 3
    key="cols_slider"
)
with st.sidebar:
    #st.markdown("## 🔧 Powered By")
    st.image("./photos/logo2.png", width=400)
    #st.image("banner.png", width=100)
    #st.image("static/streamlit_logo.png", width=100)
    st.markdown("## 🔧 Powered By: NEO4J")
    



tab1, tab2 = st.tabs(["🔍 Recommend by Text", "🖼️ Recommend by Poster"]) # Shorter tab name

with tab1:
    st.header("Describe Your Desired Movie")
    text_query = st.text_area(
        "E.g., 'A heartwarming animated film about friendship and adventure, perfect for families.'",
        key="text_query_input_area",
        value=st.session_state.last_text_query,
        height=80  # Reduced height
    )

    if st.button("Get Text-Based Recommendations", key="text_submit_btn", type="primary"):
        if text_query:
            st.session_state.last_text_query = text_query
            st.session_state.text_recommendations_detailed = []
            st.session_state.image_recommendations_detailed = []
            with st.spinner("CineBot is thinking... 🧠 (Might take a moment)"):
                try:
                    llm_response_text, initial_retrieved_movies = recommend_by_text(
                        text_query, top_k_retrieval=5, num_recommendations=st.session_state.cols_per_row_slider # Get N recs
                    )
                    parsed_llm_recs = parse_llm_recommendations(llm_response_text)
                    detailed_recs = map_llm_recs_to_retrieved_details(parsed_llm_recs, initial_retrieved_movies)
                    st.session_state.text_recommendations_detailed = detailed_recs
                except Exception as e:
                    logger.error(f"Error in text recommendation flow: {e}", exc_info=True)
                    st.error("Oops! Something went wrong. CineBot is a bit confused.")
        else:
            st.warning("Please enter a description to get recommendations.")

    if st.session_state.text_recommendations_detailed:
        st.markdown("---")
        st.subheader("CineBot's Picks For You (Text-Based):") # Changed to subheader
        display_recommendation_cards_v2(st.session_state.text_recommendations_detailed)
    # Simplified retry: user can just click the button again if needed.

with tab2:
    st.header("Find Movies by Poster Likeness")
    uploaded_image = st.file_uploader(
        "Upload a movie poster image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader_widget"
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption=f"Your query poster: {uploaded_image.name}", width=150) # Reduced width

        if st.button("Get Image-Based Recommendations", key="image_submit_btn", type="primary"):
            st.session_state.last_image_filename = uploaded_image.name
            st.session_state.text_recommendations_detailed = []
            st.session_state.image_recommendations_detailed = []
            
            with st.spinner("CineBot is analyzing the poster... 🎨 (Might take a moment)"):
                try:
                    image_bytes = uploaded_image.getvalue()
                    llm_response_text, initial_retrieved_movies = recommend_by_poster_image(
                        image_bytes, top_k_retrieval=5, num_recommendations=st.session_state.cols_per_row_slider # Get N recs
                    )
                    parsed_llm_recs = parse_llm_recommendations(llm_response_text)
                    detailed_recs = map_llm_recs_to_retrieved_details(parsed_llm_recs, initial_retrieved_movies)
                    st.session_state.image_recommendations_detailed = detailed_recs
                except Exception as e:
                    logger.error(f"Error in image recommendation flow: {e}", exc_info=True)
                    st.error("Oops! Something went wrong. CineBot needs its glasses.")

    if st.session_state.image_recommendations_detailed:
        st.markdown("---")
        st.subheader("CineBot's Picks For You (Image-Based):") # Changed to subheader
        display_recommendation_cards_v2(st.session_state.image_recommendations_detailed)

# --- Footer ---
st.markdown("---") # Use markdown for a standard hr
st.markdown("<p style='text-align: center; font-size: small;'>Powered by Neo4j | OpenAI | CLIP | Streamlit</p>", unsafe_allow_html=True)

logger.info("Streamlit app initialized/reloaded with compact layout.")