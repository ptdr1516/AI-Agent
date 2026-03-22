"""
conftest.py — adds the backend root directory to sys.path so that
all `from agent.xxx`, `from core.xxx`, and `from main import app`
imports resolve correctly regardless of which directory pytest is
launched from.
"""
import sys
import os

# Insert the backend root (the directory this file lives in) at the
# front of sys.path before any test is collected or imported.
sys.path.insert(0, os.path.dirname(__file__))
