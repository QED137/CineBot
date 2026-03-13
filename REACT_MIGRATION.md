# CineBot - Migration to React Frontend

## Quick Start Guide

This guide will help you run CineBot with the new React frontend.

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Step 1: Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install React Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root with:

```env
# Flask
FLASK_SECRET_KEY=your-secret-key-here

# Neo4j
NEO4J_URI=your-neo4j-uri
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# OpenAI
OPENAI_API_KEY=your-openai-api-key
```

### Step 4: Start the Backend (Flask API)

```bash
# From project root
python app.py
```

The Flask API will be available at `http://localhost:5000`

### Step 5: Start the Frontend (React)

In a **separate terminal**:

```bash
cd frontend
npm run dev
```

The React app will be available at `http://localhost:3000` (or `localhost:5173`)

## What Changed?

### Before (Old Frontend)
- Simple HTML templates with Flask rendering
- Vanilla JavaScript for interactivity
- Server-side rendering

### After (New React Frontend)
- Modern React 18 with hooks
- TailwindCSS for styling
- Component-based architecture
- Better state management
- Improved user experience

## Features

### 1. Chat Recommender
- Conversational interface
- Context-aware responses
- Follow-up questions supported
- Smooth animations and loading states

### 2. Poster Search
- Drag & drop file upload
- Image preview
- Visual similarity search
- Results displayed in cards

### 3. Modern UI/UX
- Responsive design
- Dark theme
- Smooth transitions
- Skeleton loading states
- Error handling

## Development Tips

### Hot Reload
Both Flask (in debug mode) and React (with Vite) support hot reload:
- Flask: Automatically reloads on Python file changes
- React: Instantly reflects component changes

### API Proxy
During development, Vite proxies `/api/*` requests to Flask backend (configured in `vite.config.js`)

### Debugging
- React: Use React DevTools browser extension
- Flask: Check console logs and use `logger.info()` or `logger.error()`

## Production Deployment

### Build React App
```bash
cd frontend
npm run build
```

This creates a `dist/` folder with optimized static files.

### Option 1: Serve React from Flask
Move the `dist/` contents to Flask's `static/` folder and update routes.

### Option 2: Separate Servers (Recommended)
- Deploy Flask as an API server
- Deploy React build to a CDN or static hosting (Vercel, Netlify, etc.)
- Configure CORS properly for production domains

## Troubleshooting

### CORS Errors
Make sure `flask-cors` is installed and Flask is running with CORS enabled.

### Port Conflicts
If ports 3000/5173 or 5000 are in use:
- React: Change port in `vite.config.js`
- Flask: Change port in `app.py` (last line)

### API Not Found
Ensure Flask backend is running and the proxy is configured in `vite.config.js`

## File Structure

```
CineBot/
├── frontend/              # React application
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── services/     # API services
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── core/                 # Backend logic
├── templates/            # Old HTML templates (can be removed)
├── static/              # Old static files (can be removed)
├── app.py               # Flask API server
└── requirements.txt
```

## Next Steps

1. **Test thoroughly**: Make sure all features work in the React version
2. **Remove old files**: Once confident, remove `templates/` and old `static/` files
3. **Optimize**: Add lazy loading, code splitting, etc.
4. **Deploy**: Set up production deployment

## Need Help?

Check the logs:
- Flask: Terminal where `python app.py` is running
- React: Terminal where `npm run dev` is running
- Browser: Developer Console (F12)
