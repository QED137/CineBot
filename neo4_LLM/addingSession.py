import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jChatMessageHistory
from uuid import uuid4
from config import settings

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

chat_llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY
)

graph = Neo4jGraph(
    url=settings.NEO4J_URI,
    username=settings.NEO4J_USERNAME,
    password=settings.NEO4J_PASSWORD
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

chat_chain = prompt | chat_llm | StrOutputParser()

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Bells", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

# while (question := input("> ")) != "exit":
    
#     response = chat_with_message_history.invoke(
#         {
#             "context": current_weather,
#             "question": question,
            
#         }, 
#         config={
#             "configurable": {"session_id": SESSION_ID}
#         }
#     )
    
    #print(response)
    
    
### this is very important for the sake of conversation. graphchain remeber session and messages    

#conversation history graphs

result = graph.query("""
                         MATCH (s:Session)-[:LAST_MESSAGE]->(last:Message)<-[:NEXT*]-(msg:Message)
                         RETURN s, last, msg
                         """)
#print(result)    

result = graph.query(
    """
    MATCH (s:Session)-[:LAST_MESSAGE]->(last:Message)
WHERE s.id = 'your session id'
MATCH p = (last)<-[:NEXT*]-(msg:Message)
UNWIND nodes(p) as msgs
RETURN DISTINCT msgs.type, msgs.content
    """
)
print(result)

