"""
Test script to verify the "who directed Salome" issue is fixed.

This script simulates:
1. User asks for romantic movie recommendations
2. System recommends movies including "Salome"
3. User asks "who directed Salome"
4. System should now correctly identify and answer the question
"""

import sys
sys.path.insert(0, '/workspaces/CineBot')

from core.core_rag import process_query
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_salome_conversation():
    """Simulate the conversation flow that was failing."""
    
    print("=" * 80)
    print("TEST: Salome Director Query Fix")
    print("=" * 80)
    
    # Step 1: Ask for romantic movies
    print("\n[USER]: suggest me some romantic movie")
    chat_history = []
    
    response1, context1 = process_query(
        user_query="suggest me some romantic movie",
        chat_history=chat_history
    )
    
    print(f"\n[BOT]: {response1}")
    print(f"\n[CONTEXT MOVIES]: {[m.get('title') for m in context1]}")
    
    # Update chat history
    chat_history.append({"role": "user", "content": "suggest me some romantic movie"})
    chat_history.append({"role": "assistant", "content": response1, "context": context1})
    
    # Step 2: Find a movie from the recommendations (or use a specific one)
    # Check if any movie with "Salome" is in the recommendations
    salome_movie = None
    for movie in context1:
        if 'salome' in movie.get('title', '').lower():
            salome_movie = movie
            break
    
    if salome_movie:
        movie_to_ask = salome_movie.get('title')
        print(f"\n✅ Found '{movie_to_ask}' in recommendations")
    else:
        # If Salome not in results, use the first movie for testing
        if context1:
            movie_to_ask = context1[0].get('title')
            print(f"\n⚠️ Salome not in recommendations, testing with '{movie_to_ask}' instead")
        else:
            print("\n❌ No movies in recommendations!")
            return
    
    # Step 3: Ask who directed the movie
    follow_up_query = f"who directed {movie_to_ask}"
    print(f"\n[USER]: {follow_up_query}")
    
    response2, context2 = process_query(
        user_query=follow_up_query,
        chat_history=chat_history
    )
    
    print(f"\n[BOT]: {response2}")
    
    # Step 4: Verify the response is not "I don't know"
    print("\n" + "=" * 80)
    print("TEST RESULTS:")
    print("=" * 80)
    
    if "don't know" in response2.lower() or "don't have" in response2.lower():
        print("❌ FAIL: Bot still says it doesn't know the answer")
        print(f"   Response: {response2}")
    elif "directed" in response2.lower() or "director" in response2.lower():
        print("✅ PASS: Bot provided director information")
        print(f"   Response: {response2}")
    else:
        print("⚠️ UNCERTAIN: Response doesn't clearly indicate success or failure")
        print(f"   Response: {response2}")

if __name__ == "__main__":
    try:
        test_salome_conversation()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        print(f"\n❌ TEST ERROR: {e}")
