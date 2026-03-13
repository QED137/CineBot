#!/usr/bin/env python3
"""
Clear all data from Neo4j database
WARNING: This will delete ALL nodes and relationships!
"""
from graph_db.connection import get_driver, close_driver
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_database():
    """Clear all nodes and relationships from the database"""
    
    driver = get_driver()
    
    logger.info("="*60)
    logger.warning("[WARNING]  WARNING: This will delete ALL data from the database!")
    logger.info("="*60)
    
    with driver.session(database="neo4j") as session:
        # Get current counts
        result = session.run("MATCH (n) RETURN count(n) AS count")
        node_count = result.single()["count"]
        
        result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
        rel_count = result.single()["count"]
        
        logger.info(f"Current database stats:")
        logger.info(f"  - Nodes: {node_count}")
        logger.info(f"  - Relationships: {rel_count}")
        
        if node_count == 0:
            logger.info("Database is already empty!")
            close_driver()
            return
        
        # Delete all nodes and relationships
        logger.info("\nDeleting all nodes and relationships...")
        session.run("MATCH (n) DETACH DELETE n")
        
        # Verify deletion
        result = session.run("MATCH (n) RETURN count(n) AS count")
        remaining = result.single()["count"]
        
        if remaining == 0:
            logger.info("[OK] Database cleared successfully!")
        else:
            logger.error(f"Failed to clear database. {remaining} nodes remaining.")
    
    close_driver()

if __name__ == "__main__":
    clear_database()
