"""
ASL Real-Time Translator - Web Application Launcher
Run with: python run_web.py
"""
import os
import sys
import webbrowser
import threading
import time

def open_browser(port):
    time.sleep(1.5)
    print(f"\n[+] Opening web interface in your browser: http://127.0.0.1:{port}\n")
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == "__main__":
    from app import app
    port = int(os.environ.get('PORT', 5000))
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
