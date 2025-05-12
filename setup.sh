#!/bin/bash

# setup.sh — one-time Codespace initializer for Streamlit + secrets

echo 
"✅ Starting setup in: $(pwd)"

# Step 1: Ensure ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  echo "🔧 Adding ~/.local/bin to PATH..."
  echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
  export PATH=$PATH:$HOME/.local/bin
fi

# Step 2: Install dependencies (if needed)
echo "📦 Installing Python packages..."
pip install --user -r requirements.txt || pip install --user streamlit

# Step 3: Generate secrets.toml
if [ -f utils/create_streamlit_env.py ]; then
  echo "🔐 Writing Streamlit secrets..."
  python3 -c "from utils.create_streamlit_env import write_streamlit_secrets_from_env; write_streamlit_secrets_from_env()"
else
  echo "⚠️ Warning: utils/create_streamlit_env.py not found — skipping secrets setup"
fi

# Step 4: Optionally run Streamlit
read -p "🚀 Do you want to launch Streamlit now? (y/n) " choice
if [[ "$choice" == [Yy]* ]]; then
  python3 -m streamlit run app.py
else
  echo "✅ Setup complete. You can run: python3 -m streamlit run app.py"
fi
