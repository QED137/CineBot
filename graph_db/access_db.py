from config import settings
import os
from langchain_community.graphs import Neo4jGraph
uri = settings.NEO4J_URI
username= settings.NEO4J_USERNAME
password = settings.NEO4J_PASSWORD
database = settings.NEO4J_DATABASE
# Initialize the Neo4jGraph instance
# Create a Neo4jGraph instance
kg = Neo4jGraph(uri, username, password, database)
cypher = """
  MATCH (n)
  RETURN count(n)
  """
  

def main():
    print("Neo4j Graph Database Connection Test")
    # Test the connection
    result = kg.query(cypher)
    if result:
        print("Connection successful!")
        print(result)
    else:
        print("Connection failed!")

    # Close the connection
       

       
    
    

if __name__ == "__main__":
    main()
    
    
        