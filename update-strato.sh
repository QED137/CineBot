#!/bin/bash

###############################################
# CineBot Strato Clean Deployment Script
# Handles conflicts and ensures clean restart
###############################################

set -e

echo " CineBot Deployment Starting..."
echo ""

# Configuration
REPO_DIR="/opt/cinebot/CineBot"
BRANCH="docker-deploy-4gb"
COMPOSE_FILE="docker-compose.4gb.yml"

# Step 1: Navigate to repository
echo " Step 1: Navigating to $REPO_DIR"
cd "$REPO_DIR" || { echo " Failed to find $REPO_DIR"; exit 1; }
echo " In repository directory"
echo ""

# Step 2: Pull latest code
echo " Step 2: Pulling latest code from GitHub"
git fetch origin
git pull origin "$BRANCH" || { echo " Git pull failed"; exit 1; }
echo " Code updated"
echo ""

# Step 3: Stop all containers
echo " Step 3: Stopping all CineBot containers"
docker stop cinebot-backend cinebot-frontend cinebot-neo4j 2>/dev/null || true
echo " Containers stopped"
echo ""

# Step 4: Remove old containers
echo "  Step 4: Removing old containers"
docker rm cinebot-backend cinebot-frontend cinebot-neo4j cinebot-redis 2>/dev/null || true
echo " Old containers removed"
echo ""

# Step 5: Clean up networks
echo " Step 5: Cleaning up networks"
docker-compose -f "$COMPOSE_FILE" down 2>/dev/null || true
echo " Networks cleaned"
echo ""

# Step 6: Remove dangling images (optional, saves space)
echo "  Step 6: Removing dangling images"
docker image prune -f
echo " Cleanup complete"
echo ""

# Step 7: Build fresh containers
echo "  Step 7: Building containers (this may take 2-3 minutes)"
docker-compose -f "$COMPOSE_FILE" build --no-cache
echo " Build complete"
echo ""

# Step 8: Start containers
echo " Step 8: Starting all services"
docker-compose -f "$COMPOSE_FILE" up -d
echo " Services started"
echo ""

# Step 9: Wait for services to initialize
echo "⏳ Step 9: Waiting for services to initialize..."
sleep 10
echo " Services initialized"
echo ""

# Step 10: Check status
echo " Step 10: Checking container status"
docker-compose -f "$COMPOSE_FILE" ps
echo ""

# Step 11: Verify health
echo " Step 11: Checking application health"
BACKEND_RUNNING=$(docker ps --filter "name=cinebot-backend" --filter "status=running" -q)
FRONTEND_RUNNING=$(docker ps --filter "name=cinebot-frontend" --filter "status=running" -q)
NEO4J_RUNNING=$(docker ps --filter "name=cinebot-neo4j" --filter "status=running" -q)

if [ -n "$BACKEND_RUNNING" ] && [ -n "$FRONTEND_RUNNING" ] && [ -n "$NEO4J_RUNNING" ]; then
    echo " All containers are running!"
else
    echo "  Warning: Some containers may not be running"
    echo ""
    echo " Showing logs for debugging:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
fi
echo ""

# Success
echo "╔════════════════════════════════════════╗"
echo "║   Deployment Successful!            ║"
echo "║   CineBot is now live               ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo " View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo " Check status: docker-compose -f $COMPOSE_FILE ps"
echo " Stop services: docker-compose -f $COMPOSE_FILE down"
echo ""
