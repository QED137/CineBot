#!/usr/bin/env python3
"""Check if environment variables are set"""
import os
from config import settings

print("Checking environment variables...\n")

variables = {
    "NEO4J_URI": settings.NEO4J_URI,
    "NEO4J_USERNAME": settings.NEO4J_USERNAME,
    "NEO4J_PASSWORD": settings.NEO4J_PASSWORD,
    "TMDB_API_KEY": settings.TMDB_API_KEY,
    "OPENAI_API_KEY": settings.OPENAI_API_KEY,
}

for name, value in variables.items():
    if value:
        if "KEY" in name or "PASSWORD" in name:
            # Show only first/last 4 chars for security
            masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"[OK] {name}: {masked}")
        else:
            print(f"[OK] {name}: {value}")
    else:
        print(f"[ERROR] {name}: NOT SET")

print("\n" + "="*80)
if not settings.OPENAI_API_KEY:
    print("[WARNING]  WARNING: OPENAI_API_KEY is not set!")
    print("\nTo fix this:")
    print("1. Go to GitHub Codespaces Secrets")
    print("2. Add OPENAI_API_KEY with your OpenAI API key")
    print("3. Restart the Codespace")
    print("\nOr create a .env file with:")
    print("OPENAI_API_KEY=sk-your-key-here")
else:
    print("[OK] All required variables are set")
