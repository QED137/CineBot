#!/bin/bash

##############################################
# CineBot Quick Reload Script
# Use this to pull changes and reload without full rebuild
# Good for: config changes, minor code updates
##############################################

set -e

echo "⚡ Quick reload mode (no rebuild)..."

# Navigate to CineBot directory
cd /opt/cinebot/CineBot

# Pull latest changes
echo "⬇️  Pulling latest changes..."
git pull origin docker-deploy-4gb

# Restart containers to reload code
echo "🔄 Restarting containers..."
docker-compose -f docker-compose.4gb.yml restart

# Wait for services
sleep 5

# Show status
echo "✅ Quick reload completed!"
echo ""
echo "📋 Container status:"
docker-compose -f docker-compose.4gb.yml ps

echo ""
echo "💡 If you added new dependencies, use ./deploy-strato.sh instead"
