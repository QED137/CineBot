from thefuzz import process, fuzz
from typing import List, Optional, Tuple
import logging

log = logging.getLogger(__name__)


# Preload movie titles from DB or a file if performance is critical.
# KNOWN_MOVIE_TITLES = ["The Matrix", "Inception", "Pulp Fiction", ...]


def find_best_match(query: str, choices: List[str], score_cutoff: int = 80) -> Optional[str]:
    """
    Finds the best fuzzy match for a query string within a list of choices.

    Args:
        query: The user's input string (e.g., movie title).
        choices: A list of valid strings to match against (e.g., all movie titles in DB).
        score_cutoff: The minimum similarity score (0-100) to consider a match.

    Returns:
        The best matching string from choices if score is above cutoff, otherwise None.
    """
    if not query or not choices:
        return None

    try:
        # process.extractOne returns a tuple: (best_match, score)
        best_match, score = process.extractOne(query, choices, scorer=fuzz.WRatio) # WRatio often works well

        log.debug(f"Fuzzy match for '{query}': Best='{best_match}', Score={score}")

        if score >= score_cutoff:
            log.info(f"Fuzzy match found: '{query}' -> '{best_match}' (Score: {score})")
            return best_match
        else:
            log.info(f"No fuzzy match found for '{query}' above cutoff {score_cutoff} (Best: '{best_match}', Score: {score})")
            return None
    except Exception as e:
        log.error(f"Error during fuzzy matching for '{query}': {e}")
        return None

#Example usage testing:
if __name__ == "__main__":
    movie_list = ["The Matrix", "The Matrix Reloaded", "The Matrix Revolutions", "Inception", "Interstellar"]
    query1 = "matrix"
    query2 = "incetion"
    query3 = "Star Wars" # Not in list

    match1 = find_best_match(query1, movie_list)
    print(f"Match for '{query1}': {match1}") # Should be 'The Matrix'

    match2 = find_best_match(query2, movie_list)
    print(f"Match for '{query2}': {match2}") # Should be 'Inception'

    match3 = find_best_match(query3, movie_list)
    print(f"Match for '{query3}': {match3}") # Should be None