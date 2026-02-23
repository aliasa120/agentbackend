#!/bin/bash
set -e

echo "🚀 Starting deployment of Agent Backend..."

# 1. Clone the repository (or pull latest if it exists)
if [ -d "agentbackend" ]; then
    echo "📂 Directory 'agentbackend' exists. Pulling latest changes..."
    cd agentbackend
    git pull origin main
else
    echo "📥 Cloning repository..."
    git clone https://github.com/aliasa120/agentbackend.git
    cd agentbackend
fi

# 2. Convert .env.example to .env if .env doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
else
    echo "✅ .env file already exists."
fi

# 3. Stop and remove existing container if it exists
if [ "$(sudo docker ps -aq -f name=my-agent-backend)" ]; then
    echo "🛑 Stopping and removing existing container..."
    sudo docker stop my-agent-backend || true
    sudo docker rm my-agent-backend || true
fi

# 4. Build the Docker image
echo "🐳 Building Docker image..."
sudo docker build -t agentbackend .

# 5. Run the container
echo "▶️ Running Docker container on port 2024..."
sudo docker run -d --name my-agent-backend -p 2024:2024 --env-file .env agentbackend

echo "✅ Deployment complete! The backend is running on port 2024."
echo "⚠️  NOTE: If you haven't filled in your actual API keys in the .env file yet, please do so using 'nano .env' and then re-run this script."
