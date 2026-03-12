# CineBot Reading Suggestions / Cinema Critique Feature Plan

This document consolidates the suggested changes for adding a **reading suggestions / cinema critique** feature to CineBot.

The goal is to let CineBot return not only movie recommendations, but also:

- curated essay suggestions
- cinema criticism links
- mixed results: movies + readings
- richer follow-up conversations based on both movies and readings

---

## 1. Product direction

CineBot should evolve from a movie recommender into a **movie + reading companion**.

Instead of only answering:

- what should I watch?

it should also answer:

- what should I read to understand this film, director, or theme?
- give me criticism on this director
- suggest essays like Taste of Cinema / Criterion / BFI / Senses of Cinema
- recommend both films and readings together

### Good search examples

- `films like Persona and essays about identity and silence`
- `readings on Tarkovsky and spirituality`
- `movies and criticism about film noir`
- `give me essays on loneliness in Wong Kar-wai`

---

## 2. Recommended architecture

Do **not** mix essays into the existing movie-only logic.

Instead, add a parallel retrieval path.

### Current retrieval lanes

- `graph_search` → factual movie/person queries
- `vector_search` → semantic movie recommendations
- `follow_up` → conversational continuation
- `poster_search` → poster-based movie retrieval

### New retrieval lanes

- `critique_search` → reading suggestions / essays / criticism
- `mixed_search` → both movies and readings

This keeps the architecture cleaner and avoids destabilizing the current movie recommendation flow.

---

## 3. Data model: use JSON as seed, not final database

A local `essays_seed.json` file is useful for the MVP.

It is **static by itself**, but that is okay initially.

### Best use of JSON

Use JSON as:

- a starter content file
- a manually curated source file
- an import/export format
- a seed dataset for Neo4j

Use **Neo4j + vector index** as the real runtime database.

### Flow

`essays_seed.json`  
→ `load_essays.py`  
→ store essays in Neo4j  
→ CineBot retrieves from Neo4j at runtime

So the JSON is not the long-term live database. It is the starter source.

---

## 4. Where to get reading material

The strongest starter sources are:

### Core trusted sources

- **Senses of Cinema**
- **BFI / Sight and Sound**
- **Criterion Current**

### Discovery layer

- **Taste of Cinema**

### Best collection categories

Start with these 4 categories:

1. director essays
2. film-specific essays
3. movement/history essays
4. theme-based criticism

### Suggested first dataset size

Start with **50–100 essays** manually curated.

That is enough for a working first version.

### Best metadata to store

For each reading item:

- title
- source
- URL
- author
- summary
- tags
- related film(s)
- related director/person
- movement
- themes

Do not start by storing full scraped article text unless you really need it.

For the first version, metadata + summary + embedding is enough.

---

## 5. Seed file structure: `essays_seed.json`

Suggested example structure:

```json
[
  {
    "essay_id": "essay_0001",
    "title": "Why Tarkovsky's Cinema Feels Spiritual",
    "source": "Senses of Cinema",
    "url": "https://example.com/tarkovsky-spiritual",
    "author": "Example Author",
    "published_date": "2025-01-10",
    "summary": "An essay about time, memory, spirituality, and transcendence in Tarkovsky's cinema, with emphasis on Stalker and Mirror.",
    "tags": ["tarkovsky", "slow cinema", "spirituality", "memory"],
    "entities": ["Andrei Tarkovsky", "Stalker", "Mirror"],
    "related_films": ["Stalker", "Mirror"],
    "related_people": ["Andrei Tarkovsky"],
    "related_movements": ["Slow Cinema"],
    "themes": ["memory", "spirituality", "time"],
    "reading_level": "intermediate"
  }
]
```

### Important clarification

The seed loader version suggested earlier does **not fetch article content from the URL**.

It only reads your local JSON and stores:

- metadata
- summary
- tags
- URL
- embedding

into Neo4j.

So the `url` is stored as a reference link only.

This is good for the MVP because it is simpler, safer, and easier to maintain.

---

## 6. Neo4j schema for essays

### Node types

#### `Essay`
Properties:

- `essay_id` (unique)
- `title`
- `source`
- `url`
- `author`
- `published_date`
- `summary`
- `tags`
- `entities`
- `related_films`
- `related_people`
- `related_movements`
- `themes`
- `reading_level`
- `embedding_text`
- `embedding`

#### `Theme`
Properties:

- `name`

#### `Movement`
Properties:

- `name`

You likely already have:

- `Movie`
- `Person`

### Recommended relationships

- `(Essay)-[:ABOUT]->(Movie)`
- `(Essay)-[:MENTIONS]->(Person)`
- `(Essay)-[:DISCUSSES]->(Theme)`
- `(Essay)-[:BELONGS_TO]->(Movement)`

### Constraints

```cypher
CREATE CONSTRAINT essay_id_unique IF NOT EXISTS
FOR (e:Essay)
REQUIRE e.essay_id IS UNIQUE;

CREATE CONSTRAINT theme_name_unique IF NOT EXISTS
FOR (t:Theme)
REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT movement_name_unique IF NOT EXISTS
FOR (m:Movement)
REQUIRE m.name IS UNIQUE;
```

### Vector index

```cypher
CREATE VECTOR INDEX essay_embeddings IF NOT EXISTS
FOR (e:Essay) ON (e.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};
```

---

## 7. Loader script: `load_essays.py`

Purpose:

- load `essays_seed.json`
- create `embedding_text`
- generate embeddings
- upsert essays into Neo4j
- link essays to movies, people, themes, movements

### Recommended `embedding_text`

Do not embed only the title.

Build it from:

- title
- summary
- tags
- entities
- related films
- related people
- related movements
- themes

Example logic:

```python
embedding_text = " ".join([
    title,
    summary,
    "Tags: " + ", ".join(tags),
    "Entities: " + ", ".join(entities),
    "Related films: " + ", ".join(related_films),
    "Related people: " + ", ".join(related_people),
    "Related movements: " + ", ".join(related_movements),
    "Themes: " + ", ".join(themes),
])
```

This will retrieve much better for thematic reading queries.

---

## 8. Core backend change: move from movie-only payload to content bundle

The current backend effectively assumes the response payload is only movies.

That is too narrow once readings are added.

### Old pattern

```python
(message, context_movies, metadata)
```

### New recommended pattern

```python
(message, payload, metadata)
```

where `payload` looks like:

```python
{
    "movies": [...],
    "readings": [...],
    "themes": [...]
}
```

This gives the frontend a stable structure for:

- movie-only
- reading-only
- mixed responses

---

## 9. Suggested `core_rag.py` changes

### 9.1 Add retrieval for essays

Suggested function:

```python
def retrieve_essays_by_similarity(query_text: str, top_k: int = 5) -> List[Dict]:
    if not kg:
        return []

    query_embedding = get_text_embedding_openai(query_text)
    if not query_embedding:
        return []

    cypher_query = """
    CALL db.index.vector.queryNodes('essay_embeddings', $top_k, $query_embedding)
    YIELD node AS e, score
    RETURN e.essay_id AS essay_id,
           e.title AS title,
           e.source AS source,
           e.url AS url,
           e.summary AS summary,
           e.tags AS tags,
           score
    ORDER BY score DESC
    """

    try:
        return kg.query(cypher_query, params={
            "top_k": top_k,
            "query_embedding": query_embedding
        }) or []
    except Exception as e:
        logger.error(f"Error querying Neo4j for essay similarity: {e}")
        return []
```

### 9.2 Add essay formatting helper

```python
def format_essays_for_llm_prompt(essays: List[Dict]) -> str:
    if not essays:
        return "No relevant essays were found."

    context_parts = []
    for i, essay in enumerate(essays[:5]):
        title = html.escape(essay.get("title", "N/A"))
        source = html.escape(essay.get("source", "Unknown"))
        summary = html.escape(essay.get("summary", ""))[:300]
        tags = ", ".join(essay.get("tags", []) or [])
        essay_str = (
            f"--- Essay Index {i+1}: {title} ---\n"
            f"Source: {source}\n"
            f"Tags: {tags}\n"
            f"Summary: {summary}\n"
            f"---"
        )
        context_parts.append(essay_str)

    return "\n\n".join(context_parts)
```

### 9.3 Add `handle_critique_search()`

Purpose:

- retrieve essay suggestions
- format them
- ask LLM to recommend the best readings
- fall back cleanly if the LLM fails

### 9.4 Add `handle_mixed_search()`

Purpose:

- retrieve movies by current vector logic
- retrieve essays by essay vector index
- return both together

### 9.5 Extend intent classification

Add new possible intents:

- `critique_search`
- `mixed_search`

#### Good critique trigger patterns

- essay
- essays
- critique
- criticism
- analysis
- reading
- readings
- article
- articles
- symbolism
- meaning
- interpretation
- cinema theory
- film theory

#### Good mixed trigger patterns

- movies and essays
- films plus readings
- watch and read
- recommend movies and criticism

### 9.6 Update `process_query()`

The current `process_query()` should evolve to always return a payload like:

```python
{
    "movies": [...],
    "readings": [...],
    "themes": [...]
}
```

This is the key structural backend change.

---

## 10. Follow-up logic change in `core_rag.py`

Right now the follow-up path expects context to be just a movie list.

### Old assumption

```python
previous_context_movies = last_bot_message.get("context", [])
```

### New recommendation

```python
previous_context = last_bot_message.get("context", {})
previous_context_movies = previous_context.get("movies", [])
previous_context_readings = previous_context.get("readings", [])
```

This is important because later you may want follow-ups like:

- `give me more essays like the first one`
- `what is the source of that article?`
- `show readings about the second movie`

---

## 11. FastAPI `endpoints.py` changes

### 11.1 Accept `mode`

The `/api/chat` endpoint should accept a new form field:

```python
mode: Optional[str] = Form(None)
```

Recommended values:

- `movies`
- `readings`
- `both`

### 11.2 Change how `process_query()` is unpacked

### Old

```python
bot_response_text, context_movies, response_metadata = process_query(...)
```

### New

```python
bot_response_text, payload, response_metadata = process_query(...)
movies = payload.get("movies", [])
readings = payload.get("readings", [])
themes = payload.get("themes", [])
```

### 11.3 Store assistant context as a dict

### Old

```python
bot_message = {
    "role": "assistant",
    "content": bot_response_text,
    "context": context_movies,
}
```

### New

```python
bot_message = {
    "role": "assistant",
    "content": bot_response_text,
    "context": {
        "movies": movies,
        "readings": readings,
        "themes": themes,
    },
}
```

### 11.4 Return richer JSON response

Recommended response shape:

```python
return JSONResponse(
    {
        "message": bot_response_text,
        "movies": movies,
        "readings": readings,
        "themes": themes,
        "response_type": response_metadata.get("response_type", "recommendation"),
        "source": response_metadata.get("source", "unknown"),
        "input_mode": response_metadata.get("input_mode", "text"),
        "meta": {
            "has_movies": bool(movies),
            "has_readings": bool(readings),
            "movie_count": len(movies),
            "reading_count": len(readings),
        }
    }
)
```

---

## 12. React frontend strategy

The active production UI is the **React frontend**, not the legacy HTML interface.

So the backend should be upgraded primarily for React.

### Important conclusion

Do **not** redesign `App.jsx`.

`App.jsx` is already fine.

The reading feature belongs inside **`ChatTab.jsx`**.

---

## 13. React `ChatTab.jsx` changes

### 13.1 Add a query mode selector

Add state:

```jsx
const [queryMode, setQueryMode] = useState('both');
```

Suggested UI choices:

- Movies
- Readings
- Both

This is better than relying only on automatic intent classification.

### 13.2 Add `ReadingCard.jsx`

Create a new component for reading suggestions.

Recommended fields:

- title
- source
- summary / why
- URL

### 13.3 Store readings in assistant messages

### Old

```jsx
const assistantMessage = {
  role: 'assistant',
  content: data.message,
  movies: data.movies || [],
  metadata: {...}
};
```

### New

```jsx
const assistantMessage = {
  role: 'assistant',
  content: data.message,
  movies: data.movies || [],
  readings: data.readings || [],
  themes: data.themes || [],
  metadata: {...}
};
```

### 13.4 Build chat history with full context bundle

### Old

```jsx
const chatHistory = messages.map(msg => ({
  role: msg.role,
  content: msg.content,
  ...(msg.movies && msg.movies.length > 0 ? { context: msg.movies } : {})
}));
```

### New

```jsx
const chatHistory = messages.map(msg => ({
  role: msg.role,
  content: msg.content,
  ...((msg.movies && msg.movies.length > 0) || (msg.readings && msg.readings.length > 0)
    ? {
        context: {
          movies: msg.movies || [],
          readings: msg.readings || [],
          themes: msg.themes || [],
        }
      }
    : {})
}));
```

### 13.5 Render readings below assistant responses

`ChatTab.jsx` already renders movie cards. Add a parallel reading section.

Suggested logic:

- if `pair.assistant.movies.length > 0` → show movie grid
- if `pair.assistant.readings.length > 0` → show reading grid
- if `pair.assistant.themes.length > 0` → show chips/tags

### 13.6 Update success toast

### Old

Movie-only toast.

### New

Handle:

- movies only
- readings only
- mixed result

---

## 14. React API service changes: `services/api.js`

The service layer should send `mode` and normalize response data.

### New API contract

#### `sendTextQuery`

```javascript
sendTextQuery(query, chatHistory = [], mode = 'both', signal = null)
```

#### `sendPosterImage`

```javascript
sendPosterImage(imageFile, chatHistory = [], mode = 'movies', signal = null)
```

### Recommended normalized return shape

```javascript
{
  message: response.data.message || '',
  movies: response.data.movies || [],
  readings: response.data.readings || [],
  themes: response.data.themes || [],
  response_type: response.data.response_type || 'recommendation',
  source: response.data.source || 'unknown',
  input_mode: response.data.input_mode || 'text',
  meta: response.data.meta || {},
}
```

This makes `ChatTab` simpler and more stable.

---

## 15. Legacy HTML interface strategy

You currently have a legacy `interface/` folder with a basic HTML UI.

### Recommendation

Do not prioritize upgrading it.

Use the React frontend as the primary UI.

The legacy interface can:

- ignore `readings` for now
- or add minimal support later

This keeps the migration simple and focused.

---

## 16. Parsing helpers if needed server-side

If the backend or legacy UI needs structured parsing of LLM output, add a separate parser for readings.

### Suggested reading output format

```text
READING: Title
SOURCE: Source Name
WHY: Short explanation
```

### Suggested parser

Create `parse_llm_readings()` parallel to `parse_llm_recommendations()`.

This is helpful if the backend wants to transform raw LLM text into structured cards.

---

## 17. Recommended development order

To keep the change manageable, implement in this order:

### Step 1
Create:

- `essays_seed.json`
- `load_essays.py`
- Neo4j essay nodes + vector index

### Step 2
Add essay retrieval to `core_rag.py`:

- `retrieve_essays_by_similarity()`
- `handle_critique_search()`
- `handle_mixed_search()`

### Step 3
Update `process_query()` to return a payload bundle:

- `movies`
- `readings`
- `themes`

### Step 4
Update FastAPI `/api/chat` to accept `mode` and return richer JSON.

### Step 5
Update `services/api.js` to send `mode` and parse richer response fields.

### Step 6
Update `ChatTab.jsx`:

- `queryMode`
- `ReadingCard`
- render readings + themes

### Step 7
Update follow-up handling to work with `context = { movies, readings, themes }`.

---

## 18. MVP vs future versions

### MVP

- manually curated `essays_seed.json`
- metadata + summary + tags only
- no article scraping
- Neo4j vector retrieval
- React rendering for readings

### Future version

Add a second ingestion pipeline that:

- fetches article text from URLs
- extracts clean article body
- summarizes it
- stores richer text in Neo4j

But this should come later, not before the first feature works.

---

## 19. Final recommendation

The cleanest and safest path is:

- keep your movie search intact
- add essays as a parallel retrieval system
- standardize the backend around a **content bundle payload**
- make React the primary rendering target
- use JSON only as seed data, not the live final database

This avoids overengineering while giving CineBot a much richer identity:

- movie recommendation engine
- cinema reading guide
- critical taste-building companion

graviton@thinkpad:~/Workspace/gitRepo/CineBot$ 

