import threading
import time
import subprocess
import sys

from agent.controllers.booking_controller import run_fastapi
from agent.views.booking_view import main as start_streamlit

if __name__ == "__main__":
    threading.Thread(target=run_fastapi, daemon=True).start()
    time.sleep(2)  # allow backend to start
    # Launch Streamlit as a separate process so it gets a proper ScriptRunContext
    subprocess.run([sys.executable, "-m", "streamlit", "run", "agent/views/booking_view.py"]) 