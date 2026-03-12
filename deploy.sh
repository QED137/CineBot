#!/bin/bash
# Quick deployment script for testing with Docker Compose

set -e

echo "🚀 CineBot Deployment Script"
echo "=============================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and fill in your API keys:"
    echo ""
    echo "  cp .env.example .env"
    echo "  nano .env  # or use your preferred editor"
    echo ""
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Environment file found"
echo "✅ Docker installed"
echo "✅ Docker Compose installed"
echo ""

# Load environment variables
source .env

# Check required environment variables
REQUIRED_VARS=("NEO4J_URI" "NEO4J_USERNAME" "NEO4J_PASSWORD" "OPENAI_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "❌ Error: Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please update your .env file with these values"
    exit 1
fi

echo "✅ All required environment variables set"
echo ""

# Ask for deployment mode
echo "Select deployment mode:"
echo "  1) Development (with live reload)"
echo "  2) Production (optimized build)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo ""
        echo "🔨 Building and starting in DEVELOPMENT mode..."
        docker-compose up --build
        ;;
    2)
        echo ""
        echo "🔨 Building and starting in PRODUCTION mode..."
        docker-compose up -d --build
        
        echo ""
        echo "⏳ Waiting for services to be ready..."
        sleep 10
        
        # Check health
        if curl -f -s http://localhost:8000/api/health > /dev/null; then
            echo "✅ Backend is healthy!"
        else
            echo "⚠️  Backend health check failed"
        fi
        
        if curl -f -s http://localhost:80 > /dev/null; then
            echo "✅ Frontend is accessible!"
        else
            echo "⚠️  Frontend not accessible"
        fi
        
        echo ""
        echo "=============================="
        echo "🎉 Deployment Complete!"
        echo "=============================="
        echo ""
        echo "Access your app at:"
        echo "  Frontend: http://localhost"
        echo "  Backend:  http://localhost:8000"
        echo "  API Docs: http://localhost:8000/docs"
        echo "  Health:   http://localhost:8000/api/health"
        echo ""
        echo "To view logs:"
        echo "  docker-compose logs -f"
        echo ""
        echo "To stop:"
        echo "  docker-compose down"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
