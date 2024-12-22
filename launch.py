import os, toml, json, sys, socket
import streamlit.web.cli as stcli

def get_local_ip():
    """Get local network IP address"""
    try:
        # Create a socket connection to an external server to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "localhost"

def load_settings():
    """Load terminal visibility setting from settings file"""
    settings_file = os.path.join(os.path.expanduser('~'), '.chat_analyzer_settings.json')
    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            return json.load(f)
    return {"show_terminal": False}  # Default setting

def setup_streamlit_config():
    """Create or update Streamlit's config.toml file with custom settings"""
    config_dir = os.path.join(os.path.expanduser('~'), '.streamlit')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.toml')
    
    config = {
        'theme': {
            'base': 'dark'
        },
        'server': {
            'port': 8501,
            'headless': True,
            'runOnSave': True
        },
        'browser': {
            'gatherUsageStats': False
        },
        'client': {
            'showErrorDetails': False,
            'toolbarMode': 'minimal'
        },
        'global': {
            'developmentMode': False
        }
    }
    
    with open(config_path, 'w') as f:
        toml.dump(config, f)

def run_streamlit():
    # Load settings
    settings = load_settings()
    show_terminal = settings.get("show_terminal", False)

    # Get network IP and set environment variables
    network_ip = get_local_ip()
    os.environ['STREAMLIT_LOCAL_IP'] = 'localhost'
    os.environ['STREAMLIT_NETWORK_IP'] = network_ip
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'  # Add this line

    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        app_path = os.path.join(base_path, 'app.py')
        # Hide console in production when show_terminal is False
        if not show_terminal:
            import win32gui, win32con
            hwnd = win32gui.GetForegroundWindow()
            win32gui.ShowWindow(hwnd, win32con.SW_HIDE)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, 'app.py')

    # Set up environment variables
    os.environ['NLTK_DATA'] = os.path.join(base_path, 'nltk_data')
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_THEME_BASE'] = 'dark'
    
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    # Setup config
    setup_streamlit_config()

    # Set up Streamlit arguments
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--theme.base=dark",
        "--server.headless=true",
        "--server.runOnSave=true",
        "--global.developmentMode=false",
        f"--browser.serverAddress={get_local_ip()}",
        "--server.port=8501"
    ]

    sys.exit(stcli.main())

if __name__ == '__main__':
    run_streamlit()