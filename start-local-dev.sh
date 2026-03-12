#!/bin/bash
# Start local Neo4j and Redis for development

set -e

echo "🚀 Starting CineBot Local Development Environment"
echo "=================================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop first"
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if containers already exist
if docker ps -a --format '{{.Names}}' | grep -q "cinebot-neo4j"; then
    echo "📦 Existing containers found"
    read -p "Do you want to restart them? (y/n): " restart
    if [[ $restart == "y" || $restart == "Y" ]]; then
        echo "🔄 Restarting containers..."
        docker-compose -f docker-compose.dev.yml restart
    else
        echo "▶️  Starting containers..."
        docker-compose -f docker-compose.dev.yml start
    fi
else
    echo "📦 Creating new containers..."
    docker-compose -f docker-compose.dev.yml up -d
fi

echo ""
echo "⏳ Waiting for services to be ready..."
echo ""

# Wait for Neo4j
echo "Waiting for Neo4j to start (this may take 30 seconds)..."
max_attempts=30
attempt=0
while ! docker exec cinebot-neo4j cypher-shell -u neo4j -p cinebot123 "RETURN 1" > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ Neo4j failed to start in time"
        echo "Check logs with: docker logs cinebot-neo4j"
        exit 1
    fi
    echo -n "."
    sleep 2
done
echo ""
echo "✅ Neo4j is ready!"

# Wait for Redis
echo "Waiting for Redis..."
max_attempts=10
attempt=0
while ! docker exec cinebot-redis redis-cli ping > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ Redis failed to start"
        exit 1
    fi
    sleep 1
done
echo "✅ Redis is ready!"

echo ""
echo "=================================================="
echo "🎉 Local Development Environment is Ready!"
echo "=================================================="
echo ""
echo "📊 Neo4j Browser:"
echo "   URL: http://localhost:7474"
echo "   Username: neo4j"
echo "   Password: cinebot123"
echo ""
echo "📦 Redis:"
echo "   Host: localhost"
echo "   Port: 6379"
echo ""
echo "🔧 Update your .env file:"
echo "   NEO4J_URI=bolt://localhost:7687"
echo "   NEO4J_USERNAME=neo4j"
echo "   NEO4J_PASSWORD=cinebot123"
echo "   REDIS_HOST=localhost"
echo ""
echo "📝 Quick commands:"
echo "   Copy local config: cp .env.local .env"
echo "   View logs:        docker-compose -f docker-compose.dev.yml logs -f"
echo "   Stop services:    docker-compose -f docker-compose.dev.yml stop"
echo "   Remove all:       docker-compose -f docker-compose.dev.yml down -v"
echo ""
echo "🚀 Next steps:"
echo "   1. Update .env with: cp .env.local .env"
echo "   2. Load your data: python build_professional_database.py"
echo "   3. Start backend:  uvicorn app_fastapi:app --reload"
echo "   4. Start frontend: cd frontend && npm run dev"
echo ""
