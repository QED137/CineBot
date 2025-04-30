from neo4j import GraphDatabase, Driver
from config import settings
import logging
import sys
import os


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global driver variable (simple approach)
# For production, consider more robust connection pooling or management
_driver = None

def get_driver() -> Driver:
    """
    Establishes connection to Neo4j database using credentials from settings.
    Uses a simple singleton pattern for the driver.
    """
    global _driver
    if _driver is None:
        log.info(f"Initializing Neo4j driver for URI: {settings.NEO4J_URI}")
        try:
            _driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            # Check connectivity
            _driver.verify_connectivity()
            log.info("Neo4j Driver initialized successfully.")
        except Exception as e:
            log.error(f"Failed to initialize Neo4j driver: {e}")
            raise ConnectionError(f"Could not connect to Neo4j at {settings.NEO4J_URI}") from e
    return _driver

def close_driver():
    """Closes the Neo4j driver connection if it's open."""
    global _driver
    if _driver is not None:
        log.info("Closing Neo4j driver.")
        _driver.close()
        _driver = None

# Example of how to use:
def main():
    try:
        driver = get_driver()
        with driver.session(database="neo4j") as session: # Specify database if not default
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            record = result.single()
            print(f"Connected to Neo4j, found {record['node_count']} nodes.")
    except ConnectionError as ce:
        print(f"Connection Error: {ce}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        close_driver()
    
    
    
if __name__ == '__main__':
    main()
    