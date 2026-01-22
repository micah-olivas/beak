# Remote Server Setup Guide

Quick guide to deploy the Discord bot on your remote server using Docker.

## Prerequisites

- Remote server with SSH access
- Docker and Docker Compose installed

## Setup

### 1. Verify Docker

```bash
docker --version
docker-compose --version
```

If Docker isn't installed:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add your user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Upload Code

```bash
# Option A: Git
git clone <your-repo-url>
cd <repo-name>

# Option B: SCP from local machine
scp -r /path/to/project user@server:/home/user/discord-bot
```

### 3. Configure Environment

```bash
cp .env.example .env
nano .env
```

Add your credentials:
```env
DISCORD_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
```

### 4. Start Bot

```bash
docker-compose up -d
```

## Common Commands

```bash
docker-compose logs -f        # View logs
docker-compose restart        # Restart
docker-compose down           # Stop
docker-compose up -d --build  # Update & restart
```

## Troubleshooting

**Bot not responding?**
- Check logs: `docker-compose logs -f`
- Verify `.env` credentials
- Ensure bot is in Discord server with proper permissions

**Permission denied?**
```bash
sudo usermod -aG docker $USER
newgrp docker
```
