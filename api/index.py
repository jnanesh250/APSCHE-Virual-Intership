import os
import sys

# Ensure the repository root is on the Python path.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Project_folder.Epic_5_Application_building.app import app
