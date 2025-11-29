#!/bin/bash

# Spotify Big Data Analysis Dashboard Launcher

echo "🎵 Starting Spotify Big Data Analysis Dashboard..."
echo ""
echo "Dashboard will open automatically in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Run streamlit
python3 -m streamlit run dashboard.py

# If streamlit module not found, try direct command
if [ $? -ne 0 ]; then
    echo "Trying alternative launch method..."
    ~/.local/bin/streamlit run dashboard.py || ~/Library/Python/3.9/bin/streamlit run dashboard.py
fi

