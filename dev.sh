#!/bin/bash
# Quick start for development - starts both backend and frontend

echo "🎬 Starting CineBot Development Servers..."

# Start Flask in one terminal
gnome-terminal --tab --title="Flask Backend" -- bash -c "source venv/bin/activate && python app.py; exec bash"

# Wait a moment
sleep 2

# Start React in another terminal
gnome-terminal --tab --title="React Frontend" -- bash -c "cd frontend && npm run dev; exec bash"

echo "✅ Servers starting in separate terminals!"
echo "   - Flask: Check 'Flask Backend' tab"
echo "   - React: Check 'React Frontend' tab"
