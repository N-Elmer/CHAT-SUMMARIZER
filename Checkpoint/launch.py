# launch.py
import os
import sys
import streamlit.web.cli as stcli
from pathlib import Path

def run_streamlit():
    # Get the directory where the executable is located
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app.py')
    else:
        # we are running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app.py')

    # Set up environment variables
    os.environ['NLTK_DATA'] = os.path.join(base_path, 'nltk_data')
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Add the base path to system path
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    # Set up Streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.address=0.0.0.0",  # Allow external access
        "--server.port=8501",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        f"--browser.serverAddress={get_local_ip()}",  # Show correct network URL
        "--theme.base=light",
        f"--browser.favicon={os.path.join(base_path, 'icon.ico')}"
    ]

    # Run Streamlit
    sys.exit(stcli.main())

def get_local_ip():
    """Get local IP address"""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

if __name__ == '__main__':
    run_streamlit()