# [MOVIE] CineBot - Quick Reference

## Installation

### First Time Setup
```bash
# 1. Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install Node dependencies
cd frontend
npm install
cd ..

# 3. Set up environment variables
# Create .env file with your API keys (see .env.example)
```

## Running the App

### Quick Start (Both servers at once)
```bash
./start.sh
```

### Manual Start
```bash
# Terminal 1 - Backend
source venv/bin/activate
python app.py

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

## URLs
- **React App**: http://localhost:3000 or http://localhost:5173
- **Flask API**: http://localhost:5000

## Common Commands

### Backend
```bash
# Activate virtual environment
source venv/bin/activate

# Install new package
pip install package-name
pip freeze > requirements.txt

# Run Flask
python app.py

# Run with gunicorn (production)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend
```bash
# Install dependencies
cd frontend && npm install

# Development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Install new package
npm install package-name
```

## File Structure
```
frontend/
├── src/
│   ├── components/     # React components
│   ├── services/       # API calls
│   ├── App.jsx         # Main app
│   └── main.jsx        # Entry point
├── package.json        # Dependencies
└── vite.config.js      # Build config
```

## API Endpoints
- `POST /api/chat` - Send text query or poster image
- `GET /api/suggestion` - Get random suggestion

## Troubleshooting

### Port already in use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### CORS errors
- Ensure Flask is running with CORS enabled
- Check Flask console for errors
- Verify proxy in vite.config.js

### Module not found
```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

## Development Tips

### Hot Reload
- **Backend**: Flask auto-reloads on .py changes
- **Frontend**: Vite auto-reloads on component changes

### Debugging
- **Backend**: Check Flask terminal for errors
- **Frontend**: Open browser DevTools (F12)
- **Network**: Check Network tab for API calls

### State Management
- Chat history stored in React state
- Session managed server-side by Flask
- No Redux needed for now (can add later)

## Production Deployment

### Build Frontend
```bash
cd frontend
npm run build
# Output: frontend/dist/
```

### Serve with Flask
Option 1: Copy dist/ to Flask static folder
Option 2: Use separate servers (recommended)

### Environment Variables
Set these in production:
- `FLASK_SECRET_KEY`
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `OPENAI_API_KEY`
- Update CORS origins in app.py

## Git Commands
```bash
# Add React frontend
git add frontend/
git commit -m "Add React frontend"

# Ignore node_modules
echo "frontend/node_modules" >> .gitignore
echo "frontend/dist" >> .gitignore
```

## Performance Tips
- Use `npm run build` for production (minified)
- Enable caching for API responses
- Consider Redis for session storage
- Add lazy loading for images
- Implement pagination for large result sets

## Next Features to Add
- [ ] User authentication
- [ ] Favorites/watchlist
- [ ] Movie ratings
- [ ] Share recommendations
- [ ] Dark/light theme toggle
- [ ] PWA support
- [ ] Movie details modal
- [ ] Advanced filters

---
**Need more help?** Check REACT_MIGRATION.md or frontend/README.md
