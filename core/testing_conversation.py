# this file is testing for apply conversational history to the rag
import os 
from config import settings
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate




NEO4J_URI = settings.NEO4J_URI
NEO4J_USERNAME = settings.NEO4J_USERNAME
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
OPENAI_API_KEY = settings.OPENAI_API_KEY
OPENAI_ENDPOINT = settings.OPENAI_ENDPOINT or "https://api.openai.com/v1"
TMDB_API_KEY = settings.TMDB_API_KEY
OMDB_API = settings.OMDB_API

#initialising llm

#1.  the minimum code to inialize 


#llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY")) # if passowrd is saved in .env

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
response= llm.invoke("What is NEO4j")

#print(response)

#2. how to use template

template = """
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
"""

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables={"fruit"})

response = llm.invoke(template.format(fruit="apple"))

print(response)


