"""
Backend Wrapper for Automated Vector Database Construction Script (ai4bharat/MSMARCO-XI)

Allows running:
    cd backend
    python build_vector_db.py --languages hi,mr,en
"""
import os
import sys

# Add project root and scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from scripts.build_vector_db import main

if __name__ == "__main__":
    main()
