#!/bin/bash

# CineBot Startup Script
echo "🎬 Starting CineBot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start Flask backend in background
echo "🚀 Starting Flask backend on port 5000..."
python app.py &
FLASK_PID=$!

# Wait a bit for Flask to start
sleep 3

# Start React frontend
echo "⚛️  Starting React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "\n🛑 Stopping servers..."
    kill $FLASK_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup INT TERM

echo "✅ CineBot is running!"
echo "   - Flask API: http://localhost:5000"
echo "   - React App: http://localhost:3000 (or 5173)"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for background processes
wait
