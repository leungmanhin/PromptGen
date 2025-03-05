#!/usr/bin/env python3
"""
Simple script to run the NL2PLN Promptgen web application
"""
import os
import sys

if __name__ == "__main__":
    # Add the parent directory to sys.path if needed
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Import the app
    from app import create_app  # Direct import from the app.py in the same directory

    # Create the app
    app = create_app()
    
    # Run the app
    app.run(debug=True)
