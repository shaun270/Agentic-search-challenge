"""
Run with:  python run.py
Or:        uvicorn backend.main:app --reload --port 8000
"""
import os
from dotenv import load_dotenv

load_dotenv()

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
