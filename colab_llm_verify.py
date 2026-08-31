# ================================================================
# COPY-PASTE THIS ENTIRE CELL TO GOOGLE COLAB
# ================================================================
# This script does everything:
#   1. Clone repo
#   2. Install Node.js + npm
#   3. Install z-ai-web-dev-sdk IN REPO ROOT (critical!)
#   4. Create config file
#   5. Run LLM verification
# ================================================================

import os
import json

# Step 1: Clone repo (if not exists)
repo = "/content/ID-Political-Sentiment-Tracker"
if not os.path.exists(repo):
    os.system("git clone https://github.com/raynzz455/ID-Political-Sentiment-Tracker.git " + repo)

os.chdir(repo)
print(f"Working directory: {os.getcwd()}")

# Step 2: Install Node.js (Colab has it, but ensure npm works)
os.system("apt-get update -qq && apt-get install -y -qq nodejs npm > /dev/null 2>&1")
print("Node.js installed")

# Step 3: Install z-ai-web-dev-sdk IN REPO ROOT (CRITICAL!)
# node_modules must be in repo root for ESM import to work
print("Installing z-ai-web-dev-sdk in repo root...")
os.system("npm install z-ai-web-dev-sdk")
print("npm install done!")

# Verify node_modules exists
if os.path.exists("node_modules/z-ai-web-dev-sdk"):
    print("✅ z-ai-web-dev-sdk found in node_modules/")
else:
    print("❌ z-ai-web-dev-sdk NOT found! Check npm install output above.")
    raise Exception("npm install failed")

# Step 4: Create config (script auto-creates it, but let's be safe)
config = {
    "baseUrl": "https://internal-api.z.ai/v1",
    "apiKey": "Z.ai",
    "chatId": "chat-6f02bcbb-29df-486b-9c2d-b07ae8567b63",
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiYjNkMGJkYjYtYzJkZC00MmIxLTg2ZjgtODkwODQwZDFjZTQ2IiwiY2hhdF9pZCI6ImNoYXQtNmYwMmJjYmItMjlkZi00ODZiLTljMmQtYjA3YWU4NTY3YjYzIiwicGxhdGZvcm0iOiJ6YWkifQ.BJmZsmnRZLSwYZK5Jny_9chyKeMurkweJaAtWhimAgY",
    "userId": "b3d0bdb6-c2dd-42b1-86f8-890840d1ce46"
}
with open(".z-ai-config", "w") as f:
    json.dump(config, f)
print("✅ Config created: .z-ai-config")

# Step 5: Run verification
print("\n--- Running LLM Verification ---\n")
