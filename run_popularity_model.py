#!/usr/bin/env python3
"""
Wrapper to run popularity model from project root
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to project root
import os
os.chdir(project_root)

# Now run the model
from ml_models import popularity_model
popularity_model.main()

