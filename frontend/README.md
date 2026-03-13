# React Frontend Setup - README

## Overview
This is the React frontend for CineBot, built with Vite, React 18, and TailwindCSS.

## Development Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

The React app will run on `http://localhost:3000` (or `http://localhost:5173` depending on Vite version).

### 3. Start Flask Backend
In a separate terminal:
```bash
# From the project root
python app.py
```

The Flask API will run on `http://localhost:5000`.

## Architecture

### Component Structure
```
src/
├── components/
│   ├── Sidebar.jsx       # Navigation sidebar
│   ├── Header.jsx        # App header with title
│   ├── ChatTab.jsx       # Chat interface for text queries
│   ├── PosterTab.jsx     # Poster upload interface
│   └── MovieCard.jsx     # Reusable movie card component
├── services/
│   └── api.js            # API service layer
├── App.jsx               # Main app component
├── main.jsx              # Entry point
└── index.css             # Global styles
```

### Features
- **Chat Recommender**: Ask questions and get movie recommendations with conversational context
- **Poster Search**: Upload a movie poster to find visually similar films
- **Modern UI**: Built with TailwindCSS for a clean, responsive design
- **Real-time Updates**: Instant feedback with loading states and animations

## Building for Production

```bash
cd frontend
npm run build
```

This creates an optimized build in the `dist/` folder.

To preview the production build:
```bash
npm run preview
```

## Configuration

The React app uses Vite's proxy feature to forward `/api/*` requests to the Flask backend during development. This is configured in `vite.config.js`.

For production, you'll need to:
1. Build the React app
2. Configure your web server (nginx, Apache, etc.) to serve the built files
3. Set up reverse proxy to forward API requests to your Flask backend

## Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
VITE_API_URL=http://localhost:5000/api
```

## Tech Stack
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first CSS framework
- **Axios** - HTTP client for API requests

## Migrating from Old Frontend

The new React frontend replaces:
- `templates/index.html` → React components
- `static/js/main.js` → React components with state management
- `static/css/style.css` → TailwindCSS utilities

Both frontends can coexist during the transition. The old frontend is served at `/` while the React app runs on `:3000` during development.
