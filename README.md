# Graph-Powered Retrieval-Augmented Generation (RAG) for Smarter Recommendations

Description: This project demonstrates the integration of Retrieval-Augmented Generation (RAG) with 
graph databases (Neo4j) and Large Language Models (LLMs) to create an intelligent and scalable recommendation
system. By leveraging the structured nature of graph databases and the dynamic
capabilities of LLMs, the system delivers contextually relevant and accurate recommendations for movies.
<p align="center">
  <img src="./photos/newmediacinebot.png" width="600" alt="CineBot Screenshot">
</p>

# Key Features 
  * Graph Database Integration: Utilizes Neo4j to structure and query complex data relationships efficiently.
  * Retrieval-Augmented Generation: Combines external data retrieval with generative AI to reduce hallucinations and improve response accuracy.
  * Movie Recommendations: Provides movie recommendations and contextual answers based on user queries.
  * Embeddings and Similarity: Leverages semantic embeddings to find similar movies dynamically.
  * Interactive Interface: A user-friendly Streamlit-based web application for an engaging user experience.

# Technology Stack:
  * Neo4j: Graph database for storing and querying movie data.
  * Streamlit: Interactive UI framework for seamless user interaction.
  * LangChain: Framework for LLM-powered chain integrations.
  * SentenceTransformers: For generating semantic embeddings.
  * Fuzzy Matching: Enhances query accuracy for movie recommendations.
  * TMDb API: Fetches movie posters and metadata.

# How It Works:
  * Data Storage: Movies and relationships are stored in Neo4j as nodes and edges.
  * Semantic Search: Queries are processed using semantic embeddings to retrieve contextually similar movies.
  * RAG Workflow: LLMs augment responses by retrieving relevant data from Neo4j.
  * Dynamic Recommendations: Personalized recommendations and answers are generated based on user input.




# Setup Instructions:
  1. Clone the repository:

     ```
        git clone https://github.com/QED137/CineBot.git
        cd cineBoat
 2. Install Dependencies

    ```
     pip install -r requirements.txt
3. Set up your Neo4j database and .env file with credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   **Option A: Use Local Neo4j (Recommended for Development)**
   ```bash
   # Start local Neo4j + Redis with Docker
   ./start-local-dev.sh
   
   # Use local config
   cp .env.local .env
   
   # Access Neo4j Browser: http://localhost:7474
   # Login: neo4j / cinebot123
   ```
   
   **Option B: Use Neo4j Aura (Cloud)**
   - Sign up at https://neo4j.com/cloud/aura/
   - Add connection details to `.env`

4. Start Redis (if not using Docker):
   ```bash
   sudo service redis-server start
   ```

5. Run the FastAPI backend:
   ```bash
   uvicorn app_fastapi:app --reload
   ```

6. Run the React frontend (in a new terminal):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

7. Access the app at http://localhost:3000

---

## [HOME] **Local Development (No Cloud Needed!)**

Run everything locally with Docker - **no more daily Neo4j Aura restarts!**

```bash
# One command to start Neo4j + Redis
./start-local-dev.sh

# See full guide
cat LOCAL_NEO4J_SETUP.md
```

**Benefits:**
- [OK] No daily manual restarts (Aura free tier limitation)
- [OK] Better performance on your hardware (6 CPU, 12GB RAM)
- [OK] Always available, no internet required
- [OK] Full control over database configuration

---

## [DEPLOY] **Cloud Deployment**

Deploy CineBot to the cloud in minutes! See our deployment guides:

- **[Strato Server Guide](DEPLOY_STRATO.md)** - Deploy to your Strato VPS (if you have SSH access) - **FREE!**
- **[Quick Start Guide](DEPLOY_QUICK_START.md)** - Deploy in 5 minutes with Railway
- **[Full Deployment Guide](DEPLOYMENT.md)** - All cloud platforms (AWS, Azure, GCP, Railway, Render, Vercel)

### Deploy with Docker:
```bash
./deploy.sh
```

### Deploy to Strato (if you have VPS):
1. SSH to your server
2. Install Docker
3. Clone repo and run: `docker-compose up -d --build`
4. See [DEPLOY_STRATO.md](DEPLOY_STRATO.md) for full guide

### One-Click Deploy to Railway:
1. Push to GitHub
2. Import to [Railway.app](https://railway.app)
3. Add environment variables
4. Done! [DONE]

---

## [DOCS] Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [FastAPI Migration](FASTAPI_MIGRATION.md)
- [Redis Setup](REDIS_SETUP.md)
- [Production Optimization](PRODUCTION_OPTIMIZATION.md)
# Use cases :
 * Personalized Movie Recommendations: Find movies similar to your favorites.
 * Knowledge Retrieval: Ask movie-related questions like "Who directed Inception?" or "What movies came out after 2010?"
 * Interactive Learning: Understand the power of combining RAG with graph-based systems.
## License
  MIT
