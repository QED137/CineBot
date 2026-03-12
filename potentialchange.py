#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
NEO4J_DATABASE = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_embedding_text(essay: Dict[str, Any]) -> str:
    parts = [
        essay.get("title", ""),
        essay.get("summary", ""),
        "Tags: " + ", ".join(essay.get("tags", [])),
        "Entities: " + ", ".join(essay.get("entities", [])),
        "Related films: " + ", ".join(essay.get("related_films", [])),
        "Related people: " + ", ".join(essay.get("related_people", [])),
        "Related movements: " + ", ".join(essay.get("related_movements", [])),
        "Themes: " + ", ".join(essay.get("themes", [])),
    ]
    return " ".join([p for p in parts if p]).strip()


def get_embedding(client: OpenAI, text: str) -> Optional[List[float]]:
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


def create_constraints_and_indexes(driver) -> None:
    queries = [
        "CREATE CONSTRAINT essay_id_unique IF NOT EXISTS FOR (e:Essay) REQUIRE e.essay_id IS UNIQUE",
        "CREATE CONSTRAINT theme_name_unique IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT movement_name_unique IF NOT EXISTS FOR (m:Movement) REQUIRE m.name IS UNIQUE",
    ]

    with driver.session(database=NEO4J_DATABASE) as session:
        for query in queries:
            session.run(query)

        # Vector index creation
        # Adjust dimensions if you use another embedding model
        vector_index_query = """
        CREATE VECTOR INDEX essay_embeddings IF NOT EXISTS
        FOR (e:Essay) ON (e.embedding)
        OPTIONS {
          indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
          }
        }
        """
        session.run(vector_index_query)

    logger.info("Constraints and vector index ensured.")


def upsert_essay(tx, essay: Dict[str, Any], embedding: List[float], embedding_text: str) -> None:
    query = """
    MERGE (e:Essay {essay_id: $essay_id})
    SET e.title = $title,
        e.source = $source,
        e.url = $url,
        e.author = $author,
        e.published_date = $published_date,
        e.summary = $summary,
        e.tags = $tags,
        e.entities = $entities,
        e.related_films = $related_films,
        e.related_people = $related_people,
        e.related_movements = $related_movements,
        e.themes = $themes,
        e.reading_level = $reading_level,
        e.embedding_text = $embedding_text,
        e.embedding = $embedding
    """
    tx.run(
        query,
        essay_id=essay["essay_id"],
        title=essay.get("title"),
        source=essay.get("source"),
        url=essay.get("url"),
        author=essay.get("author"),
        published_date=essay.get("published_date"),
        summary=essay.get("summary"),
        tags=essay.get("tags", []),
        entities=essay.get("entities", []),
        related_films=essay.get("related_films", []),
        related_people=essay.get("related_people", []),
        related_movements=essay.get("related_movements", []),
        themes=essay.get("themes", []),
        reading_level=essay.get("reading_level"),
        embedding_text=embedding_text,
        embedding=embedding,
    )


def link_themes(tx, essay_id: str, themes: List[str]) -> None:
    query = """
    MATCH (e:Essay {essay_id: $essay_id})
    UNWIND $themes AS theme_name
    MERGE (t:Theme {name: theme_name})
    MERGE (e)-[:DISCUSSES]->(t)
    """
    tx.run(query, essay_id=essay_id, themes=themes)


def link_movements(tx, essay_id: str, movements: List[str]) -> None:
    query = """
    MATCH (e:Essay {essay_id: $essay_id})
    UNWIND $movements AS movement_name
    MERGE (m:Movement {name: movement_name})
    MERGE (e)-[:BELONGS_TO]->(m)
    """
    tx.run(query, essay_id=essay_id, movements=movements)


def link_movies(tx, essay_id: str, related_films: List[str]) -> None:
    query = """
    MATCH (e:Essay {essay_id: $essay_id})
    UNWIND $films AS film_title
    MATCH (m:Movie)
    WHERE toLower(m.title) = toLower(film_title)
       OR toLower(m.title) CONTAINS toLower(film_title)
    MERGE (e)-[:ABOUT]->(m)
    """
    tx.run(query, essay_id=essay_id, films=related_films)


def link_people(tx, essay_id: str, related_people: List[str]) -> None:
    query = """
    MATCH (e:Essay {essay_id: $essay_id})
    UNWIND $people AS person_name
    MATCH (p:Person)
    WHERE toLower(p.name) = toLower(person_name)
       OR toLower(p.name) CONTAINS toLower(person_name)
    MERGE (e)-[:MENTIONS]->(p)
    """
    tx.run(query, essay_id=essay_id, people=related_people)


def main() -> None:
    if not ESSAYS_JSON_PATH.exists():
        raise FileNotFoundError(f"Could not find {ESSAYS_JSON_PATH}")

    essays = load_json(ESSAYS_JSON_PATH)
    logger.info(f"Loaded {len(essays)} essays from JSON.")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        create_constraints_and_indexes(driver)

        with driver.session(database=NEO4J_DATABASE) as session:
            for essay in essays:
                essay_id = essay["essay_id"]
                embedding_text = build_embedding_text(essay)
                embedding = get_embedding(openai_client, embedding_text)

                if not embedding:
                    logger.warning(f"Skipping {essay_id} because embedding failed.")
                    continue

                session.execute_write(upsert_essay, essay, embedding, embedding_text)
                session.execute_write(link_themes, essay_id, essay.get("themes", []))
                session.execute_write(link_movements, essay_id, essay.get("related_movements", []))
                session.execute_write(link_movies, essay_id, essay.get("related_films", []))
                session.execute_write(link_people, essay_id, essay.get("related_people", []))

                logger.info(f"Upserted essay {essay_id}: {essay.get('title')}")

    finally:
        driver.close()

    logger.info("Essay loading complete.")


if __name__ == "__main__":
    main()