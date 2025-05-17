import streamlit as st
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the app_launcher from statistics_engine
from statistics_engine.app_launcher import main

# Run the app
if __name__ == "__main__":
    main() 