import os
import sys
import webbrowser
import threading
import time

# Ensure local config directory for matplotlib to prevent Windows permission errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.mplconfig'))

from flask import Flask, Response, render_template, jsonify, request
from sign_detector import SignLanguageDetector

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
detector = SignLanguageDetector()

def gen_frames():
    while True:
        try:
            frame = detector.get_frame()
            if frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception:
            continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_translation')
def get_translation():
    return jsonify({
        'translation': detector.get_translation(),
        'current_letter': detector.current_letter,
        'confidence': round(detector.current_confidence * 100, 1)
    })

@app.route('/add_space', methods=['POST', 'GET'])
def add_space():
    detector.add_space()
    return jsonify({'status': 'success', 'translation': detector.get_translation()})

@app.route('/backspace', methods=['POST', 'GET'])
def backspace():
    detector.backspace()
    return jsonify({'status': 'success', 'translation': detector.get_translation()})

@app.route('/reset', methods=['POST', 'GET'])
def reset():
    detector.reset()
    return jsonify({'status': 'reset', 'translation': ''})

def open_browser(port):
    time.sleep(1.5)
    print(f"\n[+] Opening web interface in your default browser: http://127.0.0.1:{port}\n")
    webbrowser.open(f"http://127.0.0.1:{port}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 65)
    print("  AMERICAN SIGN LANGUAGE (ASL) REAL-TIME TRANSLATOR")
    print("  Web Server Mode")
    print(f"  Access locally at: http://127.0.0.1:{port}")
    print("=" * 65)
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

