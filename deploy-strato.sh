#!/bin/bash

##############################################
# CineBot Strato Deployment Script
# Run this on your Strato server to update the app
##############################################

set -e  # Exit on any error

echo "🚀 Starting CineBot deployment..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REPO_DIR="/opt/cinebot/CineBot"
BRANCH="docker-deploy-4gb"
COMPOSE_FILE="docker-compose.4gb.yml"

# Navigate to repo directory
echo -e "${YELLOW}📂 Navigating to ${REPO_DIR}...${NC}"
cd "$REPO_DIR" || { echo -e "${RED}❌ Failed to navigate to $REPO_DIR${NC}"; exit 1; }

# Pull latest changes
echo -e "${YELLOW}⬇️  Pulling latest changes from ${BRANCH}...${NC}"
git fetch origin
git pull origin "$BRANCH"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to pull changes${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Successfully pulled latest changes${NC}"

# Stop all CineBot containers forcefully
echo -e "${YELLOW}🛑 Stopping all CineBot containers...${NC}"
docker stop $(docker ps -q --filter name=cinebot) 2>/dev/null || echo "No running containers to stop"

# Remove containers and clean up
echo -e "${YELLOW}🧹 Cleaning up old containers...${NC}"
docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true

# Rebuild and start containers
echo -e "${YELLOW}🏗️  Rebuilding and starting containers...${NC}"
docker-compose -f "$COMPOSE_FILE" up -d --build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start containers${NC}"
    exit 1
fi

# Wait for containers to initialize
echo -e "${YELLOW}⏳ Waiting for services to start...${NC}"
sleep 8

# Check container status
echo -e "${YELLOW}📋 Container status:${NC}"
docker-compose -f "$COMPOSE_FILE" ps

# Check if all containers are running
RUNNING=$(docker-compose -f "$COMPOSE_FILE" ps | grep -c "Up" || true)
if [ "$RUNNING" -lt 2 ]; then
    echo -e "${RED}⚠️  Warning: Not all containers are running!${NC}"
    echo -e "${YELLOW}📜 Error logs:${NC}"
    docker-compose -f "$COMPOSE_FILE" logs --tail=100
    exit 1
fi

# Show recent logs
echo -e "${YELLOW}📜 Recent logs:${NC}"
docker-compose -f "$COMPOSE_FILE" logs --tail=30

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🎬 CineBot is now running with the latest changes${NC}"
echo ""
echo "💡 View live logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "💡 Check status: docker-compose -f $COMPOSE_FILE ps"
