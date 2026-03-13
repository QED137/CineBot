# CineBot Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                          │
│                  http://localhost:3000                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTP Requests
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND                            │
│                     (Vite + React 18)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Components                                           │  │
│  │  ├── Sidebar.jsx      (Navigation)                    │  │
│  │  ├── Header.jsx       (App header)                    │  │
│  │  ├── ChatTab.jsx      (Text queries)                  │  │
│  │  ├── PosterTab.jsx    (Image upload)                  │  │
│  │  └── MovieCard.jsx    (Movie display)                 │  │
│  │                                                        │  │
│  │  Services                                             │  │
│  │  └── api.js          (Axios HTTP client)              │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ /api/* requests
                         │ (proxied by Vite)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND                             │
│                  http://localhost:5000                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Endpoints (app.py)                              │  │
│  │  ├── POST /api/chat                                   │  │
│  │  └── GET  /api/suggestion                            │  │
│  │                                                        │  │
│  │  Middleware                                           │  │
│  │  ├── CORS (flask-cors)                                │  │
│  │  └── Session (flask-session)                         │  │
│  └──────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 CORE RAG ENGINE (core_rag.py)               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Intent Classification                                │  │
│  │  ├── graph_search    (Cypher queries)                 │  │
│  │  ├── vector_search   (Embeddings)                     │  │
│  │  └── follow_up       (Context-aware)                  │  │
│  │                                                        │  │
│  │  Processing Pipeline                                  │  │
│  │  ├── Text → OpenAI Embeddings                         │  │
│  │  ├── Image → CLIP Embeddings                          │  │
│  │  └── Query → LangChain → Neo4j                        │  │
│  └──────────────────────┬───────────────────────────────┘  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ├──────────────┬─────────────────┐
                           ▼              ▼                 ▼
                    ┌──────────┐   ┌──────────┐    ┌──────────┐
                    │  Neo4j   │   │  OpenAI  │    │   CLIP   │
                    │ Graph DB │   │   API    │    │  Model   │
                    └──────────┘   └──────────┘    └──────────┘
```

## Data Flow

### Text Query Flow
```
User Input → ChatTab → api.sendTextQuery() → POST /api/chat
    → process_query() → classify_intent()
    → handle_vector_search() / handle_graph_search()
    → OpenAI Embeddings → Neo4j Vector Search
    → LLM Response → MovieCard components → User
```

### Image Query Flow
```
User Upload → PosterTab → api.sendPosterImage() → POST /api/chat
    → process_query() → recommend_by_poster_image()
    → CLIP Embeddings → Neo4j Vector Search
    → LLM Response → MovieCard components → User
```

## Technology Stack

### Frontend
| Technology | Purpose | Version |
|------------|---------|---------|
| React | UI Framework | 18.3.1 |
| Vite | Build Tool | 5.3.1 |
| TailwindCSS | Styling | 3.4.4 |
| Axios | HTTP Client | 1.7.2 |

### Backend
| Technology | Purpose | Notes |
|------------|---------|-------|
| Flask | Web Framework | API server |
| Flask-CORS | CORS Support | Cross-origin requests |
| Flask-Session | Session Management | Server-side sessions |
| LangChain | LLM Framework | Text-to-Cypher |
| OpenAI | Embeddings & LLM | GPT-4 & ada-002 |

### Database & AI
| Technology | Purpose | Notes |
|------------|---------|-------|
| Neo4j | Graph Database | Vector indexes |
| CLIP | Vision Model | Poster embeddings |
| Transformers | ML Models | HuggingFace |
| PyTorch | ML Framework | CUDA support |

## Component Communication

### React State Management
```javascript
// ChatTab.jsx
const [messages, setMessages] = useState([]);
const [isLoading, setIsLoading] = useState(false);

// Flow:
1. User types message
2. Add to messages array
3. Send to API
4. Receive response
5. Update messages with response
```

### API Service Layer
```javascript
// services/api.js
export const chatAPI = {
  sendTextQuery: async (query, chatHistory) => { ... },
  sendPosterImage: async (imageFile) => { ... },
  getSuggestion: async () => { ... }
};
```

## Key Features

### 1. Intent Classification
- **graph_search**: Factual questions about movies/people
- **vector_search**: Recommendation based on mood/theme
- **follow_up**: Context-aware follow-up questions

### 2. Multi-modal Search
- **Text**: Natural language descriptions
- **Image**: Visual similarity via CLIP
- **Hybrid**: Combining both approaches

### 3. Context Management
- **Client-side**: Full chat history in React state
- **Server-side**: Last 6 turns trimmed in Flask session
- **RAG Pipeline**: Context passed to LLM for responses

## Performance Optimizations

### Frontend
- [OK] Component lazy loading (can add)
- [OK] Image lazy loading (can add)
- [OK] Debounced search (can add)
- [OK] Virtual scrolling for long lists (can add)
- [OK] Build optimization with Vite

### Backend
- [OK] Vector indexes in Neo4j
- [OK] Session trimming (max 6 turns)
- [OK] Connection pooling (can add)
- [OK] Caching layer (can add Redis)

## Security Considerations

### Current
- [OK] CORS configured for specific origins
- [OK] Session cookies with HttpOnly
- [OK] File size limits (16MB max)
- [OK] Input validation

### Recommended
- [ ] Add rate limiting
- [ ] Implement authentication (JWT)
- [ ] Add CSRF protection
- [ ] Sanitize all user inputs
- [ ] Use environment-specific CORS origins

## Deployment Architecture

### Development
```
Frontend (Vite Dev Server) :3000
    ↓ (proxy)
Backend (Flask Debug) :5000
    ↓
Neo4j, OpenAI APIs
```

### Production (Recommended)
```
Frontend (Static CDN - Vercel/Netlify)
    ↓ (HTTPS)
Backend (Gunicorn + Nginx) :5000
    ↓
Neo4j (Cloud), OpenAI API
```

## File Organization

```
CineBot/
├── frontend/                 # React SPA
│   ├── src/
│   │   ├── components/      # UI Components
│   │   ├── services/        # API Layer
│   │   ├── hooks/           # Custom hooks (future)
│   │   ├── App.jsx          # Root component
│   │   ├── main.jsx         # Entry point
│   │   └── index.css        # Global styles
│   ├── public/              # Static assets
│   ├── package.json         # Node dependencies
│   └── vite.config.js       # Build config
│
├── core/                    # Backend logic
│   ├── core_rag.py          # Main RAG engine
│   ├── cypher_generation.py # Graph queries
│   └── recommendation_service.py
│
├── embeddings/              # Embedding generators
├── graph_db/                # Neo4j connections
├── llm_integration/         # LLM chains & prompts
├── multimodal/              # CLIP & vision
├── utils/                   # Helper functions
│
├── app.py                   # Flask API server
├── requirements.txt         # Python deps
├── start.sh                 # Startup script
└── REACT_MIGRATION.md      # Migration guide
```

## Development Workflow

### Adding a New Feature

1. **Frontend**:
   ```bash
   cd frontend/src/components
   # Create NewComponent.jsx
   # Import in App.jsx or parent
   # Test with npm run dev
   ```

2. **Backend**:
   ```python
   # Add endpoint in app.py
   @app.route('/api/new-feature', methods=['POST'])
   def new_feature():
       # Logic here
       return jsonify(result)
   ```

3. **Connect**:
   ```javascript
   // In services/api.js
   export const chatAPI = {
       newFeature: async (data) => {
           return await api.post('/new-feature', data);
       }
   };
   ```

## Testing Strategy

### Frontend
- Manual testing in browser
- React DevTools for state inspection
- Network tab for API debugging

### Backend
- Flask debug mode for errors
- Logger statements in core_rag.py
- Neo4j Browser for query testing

### Integration
- Test both servers running
- Verify CORS headers
- Check session persistence

---

**Last Updated**: March 1, 2026  
**Status**: [OK] Production Ready
