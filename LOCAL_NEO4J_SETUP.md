# Running Neo4j Locally with Docker

## Why Use Local Neo4j?

[OK] **No daily manual restart** - Aura free tier pauses  
[OK] **Better performance** - Your hardware: 6 CPU, 12GB RAM, 320GB storage  
[OK] **Always available** - No internet required  
[OK] **Full control** - Configure memory, plugins, etc.  
[OK] **Faster development** - No network latency  
[OK] **Free forever** - No usage limits  

---

## Quick Start (5 minutes)

### Option 1: Docker Compose (Recommended)

**Start Neo4j + Redis:**
```bash
# Start just the databases
docker-compose -f docker-compose.dev.yml up -d

# Wait for Neo4j to be ready (30 seconds)
docker logs cinebot-neo4j -f
# Wait until you see: "Remote interface available at http://localhost:7474/"
```

**Access Neo4j Browser:**
- Open: http://localhost:7474
- Login: 
  - Username: `neo4j`
  - Password: `cinebot123`

**Update your .env file:**
```bash
# Copy local config
cp .env.local .env

# Or manually edit .env:
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=cinebot123
REDIS_HOST=localhost
```

**Start your backend:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI
uvicorn app_fastapi:app --reload
```

**Load your movie data:**
```bash
# Run your database creation script
python create_movieDB.py
# Or your preferred data loading script
```

---

### Option 2: Docker Run (Manual)

```bash
# 1. Create a network
docker network create cinebot-network

# 2. Start Neo4j
docker run -d \
  --name cinebot-neo4j \
  --network cinebot-network \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/cinebot123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_memory_heap_initial__size=2G \
  -e NEO4J_dbms_memory_heap_max__size=4G \
  -e NEO4J_dbms_memory_pagecache_size=2G \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:5.15-community

# 3. Start Redis
docker run -d \
  --name cinebot-redis \
  --network cinebot-network \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine redis-server --appendonly yes

# 4. Check if running
docker ps
```

---

## [STATS] Migrating Data from Neo4j Aura

If you have data in Neo4j Aura, migrate it to local:

### Method 1: Export/Import (Recommended)

**From Neo4j Aura:**
```cypher
// 1. In Aura Neo4j Browser (https://console.neo4j.io)
// Export your data
CALL apoc.export.cypher.all("movies-export.cypher", {
    format: "cypher-shell",
    useOptimizations: {type: "UNWIND_BATCH", unwindBatchSize: 20}
})
YIELD file, batches, source, format, nodes, relationships, properties, time
RETURN file, nodes, relationships, time;
```

**Download the export file from Aura** (check their export/backup options)

**To Local Neo4j:**
```bash
# 1. Copy export file to Neo4j import folder
docker cp movies-export.cypher cinebot-neo4j:/var/lib/neo4j/import/

# 2. Import into local Neo4j
docker exec -it cinebot-neo4j cypher-shell -u neo4j -p cinebot123 \
  "CALL apoc.cypher.runFile('/var/lib/neo4j/import/movies-export.cypher')"
```

### Method 2: Re-run Your Creation Scripts

Simply re-run your existing database creation scripts against local Neo4j:

```bash
# Make sure .env points to local Neo4j
NEO4J_URI=bolt://localhost:7687

# Run your scripts
python build_professional_database.py
# or
python create_movieDB.py
# or whatever script you used originally
```

### Method 3: neo4j-admin Dump/Load (Full backup)

**From Aura:** Download database backup from Aura console

**To Local:**
```bash
# 1. Stop Neo4j
docker-compose -f docker-compose.dev.yml stop neo4j

# 2. Place dump file in a location
# 3. Load the dump
docker run --rm \
  -v neo4j_data:/data \
  -v /path/to/your:/backups \
  neo4j:5.15-community \
  neo4j-admin database load --from-path=/backups neo4j

# 4. Restart Neo4j
docker-compose -f docker-compose.dev.yml start neo4j
```

---

## [CONFIG] Configuration

### Memory Settings (for your 12GB RAM)

The docker-compose already configures:
- **Heap Size**: 2GB initial, 4GB max
- **Page Cache**: 2GB
- **Total Neo4j**: ~6GB max

You can adjust in `docker-compose.dev.yml`:

```yaml
environment:
  - NEO4J_dbms_memory_heap_initial__size=3G  # Increase if needed
  - NEO4J_dbms_memory_heap_max__size=6G      # Increase if needed
  - NEO4J_dbms_memory_pagecache_size=3G      # Increase if needed
```

### Performance Tuning

```yaml
environment:
  # Improve write performance
  - NEO4J_dbms_checkpoint_interval_time=15m
  - NEO4J_dbms_checkpoint_interval_tx=100000
  
  # Connection pool
  - NEO4J_dbms_connector_bolt_thread__pool__max__size=400
  
  # Query timeout
  - NEO4J_dbms_transaction_timeout=30s
```

---

## [STATS] Monitoring Your Local Neo4j

### Check Status
```bash
# View logs
docker logs cinebot-neo4j -f

# Check container stats
docker stats cinebot-neo4j

# Check health
curl http://localhost:7474
```

### Neo4j Browser Queries
```cypher
// Check database size
CALL apoc.meta.stats()
YIELD nodeCount, relCount, labelCount
RETURN nodeCount, relCount, labelCount;

// Check memory usage
CALL dbms.listConfig() 
YIELD name, value 
WHERE name CONTAINS 'memory'
RETURN name, value;

// Check running queries
SHOW TRANSACTIONS;
```

---

## [BACKUP] Backup Your Local Neo4j

### Automatic Backups
```bash
# Create backup script
nano backup-neo4j.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec cinebot-neo4j neo4j-admin database dump neo4j \
  --to-path=/var/lib/neo4j/import

# Copy to host
docker cp cinebot-neo4j:/var/lib/neo4j/import/neo4j.dump \
  "$BACKUP_DIR/neo4j_backup_$DATE.dump"

echo "Backup created: $BACKUP_DIR/neo4j_backup_$DATE.dump"
```

### Schedule with Cron
```bash
# Run daily at 2 AM
crontab -e
# Add:
0 2 * * * /path/to/backup-neo4j.sh
```

---

## [DEPLOY] Development Workflow

### Daily Usage

**Start everything:**
```bash
# Start databases
docker-compose -f docker-compose.dev.yml up -d

# Start backend (in another terminal)
source .venv/bin/activate
uvicorn app_fastapi:app --reload

# Start frontend (in another terminal)
cd frontend
npm run dev
```

**Stop everything:**
```bash
# Stop backend (Ctrl+C in terminal)

# Stop databases
docker-compose -f docker-compose.dev.yml down
```

**Restart without losing data:**
```bash
docker-compose -f docker-compose.dev.yml restart
```

**Complete cleanup (deletes all data):**
```bash
docker-compose -f docker-compose.dev.yml down -v
```

---

## 🆚 Local vs Aura Comparison

| Feature | Local Docker | Neo4j Aura Free |
|---------|--------------|-----------------|
| **Cost** | Free | Free |
| **Manual Start** | Auto-starts | Daily manual restart |
| **Performance** | Fast (local) | Network latency |
| **Memory** | 6GB+ (you choose) | Limited |
| **Storage** | 320GB available | 200k nodes limit |
| **Internet Required** | No | Yes |
| **Setup Time** | 5 minutes | 10 minutes |
| **Backups** | Manual | Automatic |
| **High Availability** | No | Yes (paid) |

**For development: Local Docker wins!**  
**For production: Use Aura or hosted Neo4j**

---

##  Switching Between Local and Cloud

You can easily switch by changing `.env`:

**Local Neo4j:**
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=cinebot123
```

**Neo4j Aura:**
```bash
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_PASSWORD=your_aura_password
```

Just restart your backend after changing!

---

## 🆘 Troubleshooting

**Neo4j not starting:**
```bash
# Check logs
docker logs cinebot-neo4j

# Common issue: Port already in use
sudo lsof -i :7687
# Kill the process or stop Aura Desktop
```

**Can't connect from Python:**
```bash
# Make sure URI is correct
NEO4J_URI=bolt://localhost:7687  # not neo4j://

# Test connection
python test_neo4j_connection.py
```

**Out of memory:**
```bash
# Reduce memory in docker-compose.dev.yml
NEO4J_dbms_memory_heap_max__size=2G
NEO4J_dbms_memory_pagecache_size=1G
```

**Need to reset everything:**
```bash
# Nuclear option - deletes all data
docker-compose -f docker-compose.dev.yml down -v
docker volume prune -f
# Then start fresh
docker-compose -f docker-compose.dev.yml up -d
```

---

## [GROWTH] Resource Usage

With your specs (6 CPU, 12GB RAM, 320GB):

**What's running:**
- Neo4j: ~6GB RAM, 2-3GB disk
- Redis: ~100MB RAM, <1GB disk
- Backend: ~500MB RAM
- Frontend: Minimal

**Total:** ~7GB RAM, ~5GB disk  
**Remaining:** 5GB RAM free, 315GB disk free [OK]

You have plenty of resources!

---

## [TARGET] Recommended Setup

**For your use case, I recommend:**

1. [OK] **Use local Neo4j** (Docker) for development
2. [OK] **Use local Redis** (Docker) for caching
3. [OK] **Run backend and frontend** natively (not in Docker)
4. [CONFIG] **Deploy to cloud** with cloud Neo4j only for production

This gives you:
- Fast development
- No daily restarts
- Full control
- Easy debugging

---

## [OK] Quick Setup Checklist

- [ ] Install Docker Desktop
- [ ] Run `docker-compose -f docker-compose.dev.yml up -d`
- [ ] Access Neo4j Browser at http://localhost:7474
- [ ] Update `.env` to use `bolt://localhost:7687`
- [ ] Load your movie data
- [ ] Start backend with `uvicorn app_fastapi:app --reload`
- [ ] Verify health at http://localhost:8000/api/health
- [ ] Done! [DONE]

**You're now free from Aura's daily manual restarts!** [DEPLOY]
