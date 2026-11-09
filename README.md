# 🤟 American Sign Language (ASL) Real-Time Translator
### *Neubrutalism Edition // Universal AI Engine*

An end-to-end Computer Vision and Deep Learning system that recognizes American Sign Language (ASL) finger-spelling (A–Z) from a live webcam feed, aggregates recognized letters into words, applies automatic grammar and spell-checking, and translates them into coherent English sentences in real-time.

---

## ✨ Features

- **26-Letter ASL Alphabet Recognition:** Classifies all letters from **A to Z** using high-speed TFLite / Keras transfer-learned neural network weights.
- **🎨 Neubrutalism Design Language:** Thick `3px` solid black borders, hard `5px 5px 0px #000` block drop shadows, high-contrast retro arcade styling (High-Voltage Yellow, Electric Cyan, Mint Green, Crisp White & Pitch Black). **Zero purple guarantee.**
- **⚡ Universal Compatibility:** Runs seamlessly on both Python 3.9 and Python 3.10 (`.venv`), with built-in Keras 2 / Keras 3 / TFLite support and bundled offline MediaPipe tasks.
- **⏱️ Smart Letter Debouncing & Stabilization:** Requires holding a sign steadily for ~0.4s to register a letter (preventing repetitive letter spamming like `AAAAAA`). A visual progress bar indicates when a sign is being registered.
- **📖 In-App ASL Alphabet Cheat-Sheet:** Collapsible on-screen reference guide showing hand poses and hints for all 26 letters (A–Z).
- **Auto Sentence Builder & Grammar Correction:** Pausing between signs automatically finalizes the word and cleans up spelling/grammar.
- **Dual Presentation Modes:**
  - **🖥️ Desktop Mode (OpenCV):** Fast, standalone desktop window with interactive HUD, live progress bar, and keyboard shortcuts.
  - **🌐 Web Application Mode (Flask):** Modern Neubrutalist web dashboard featuring live video streaming, Text-to-Speech (TTS) audio playback, one-click copy, and interactive translation editing.
- **📷 Built-in Data Collector:** Expand or fine-tune your dataset anytime using the included webcam data collection utility.

---

## 📁 Project Structure

The project has been consolidated into a clean, single workspace:

```
Sign Language/
├── Model/                      # Active trained models & labels
│   ├── model.tflite            # High-speed universal TFLite model (2.1 MB)
│   ├── keras_model.h5          # 26-class Keras deep learning model (2.4 MB)
│   └── labels.txt              # Alphabet classes (A–Z)
├── templates/                  # Web interface templates
│   └── index.html              # Neubrutalism web dashboard with TTS & Cheat-Sheet
├── Data/                       # Custom-collected dataset (~9,300 images across A–Z)
│   ├── A/, B/, C/ ... Z/
├── hand_landmarker.task        # Bundled offline MediaPipe task model
├── sign_detector.py            # Core hand tracking, inference engine & direct desktop runner
├── grammar_processor.py        # Sentence construction & spelling correction processor
├── app.py                      # Flask web application server
├── desktop_app.py              # Standalone OpenCV desktop application
├── data_collection.py          # Interactive dataset recorder for new hand signs
├── run_desktop.bat / .py       # 1-click launchers for Desktop Mode
├── run_web.bat / .py           # 1-click launchers for Web Mode
├── requirements.txt            # Python package dependencies
├── datasets/                   # Reference datasets (Roboflow Darknet & Kaggle ASL)
└── archive_experiments/        # Preserved research notebooks & earlier prototypes
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
Ensure you have Python 3.9+ installed. The required packages are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Key packages: `opencv-python`, `cvzone`, `mediapipe`, `tensorflow`, `flask`, `numpy`, `textblob`)*

---

### 2. Running the Translator

You have two convenient ways to run the project:

#### Option A: Web Application Mode (Recommended)
Double-click **`run_web.bat`** or run:
```bash
python run_web.py
```
*(Or with `.venv`: `& ".venv\Scripts\python.exe" app.py`)*

* Your default web browser will automatically open to `http://127.0.0.1:5000`.
* Features:
  - 🎥 Live camera view with detection overlay and confidence metrics.
  - 🔊 **"SPEAK (TTS)" button:** Reads your translated sentence out loud using browser speech synthesis.
  - 📋 **"COPY TEXT" button:** Copies the translated sentence directly to your clipboard.
  - ␣ **"SPACE" / ⌫ "BACKSPACE" / ✕ "CLEAR" buttons** for instant editing.
  - 📖 **"TOGGLE ASL ALPHABET CHEAT-SHEET" button:** Expands a visual reference index for letters A–Z.

---

#### Option B: Standalone Desktop Mode
Double-click **`run_desktop.bat`** or run:
```bash
python sign_detector.py
```
*(Or with `.venv`: `& ".venv\Scripts\python.exe" sign_detector.py`)*

* Opens an OpenCV window with real-time HUD and visual stability progress indicator.
* **Desktop Keyboard Shortcuts:**
  | Key | Action |
  |---|---|
  | `Hold Sign` | Fills the green progress bar; registers the letter once full (~0.4s) |
  | `Space` | Adds a space / finishes current word |
  | `Backspace` or `b` | Deletes the last character |
  | `c` | Clears the entire sentence |
  | `q` or `Esc` | Quits the application |

---

### 3. Collecting More Training Data (Optional)

If you wish to record more samples for any letter:
```bash
python data_collection.py
```
1. Enter the letter you want to capture (e.g. `A`).
2. Show your hand to the camera.
3. Press **`s`** to save preprocessed, aspect-ratio normalized $300 \times 300$ images into `Data/<letter>/`.
4. Press **`q`** to finish.

---

## 🧠 How It Works (Under the Hood)

```
Webcam Frame
     │
     ▼
cvzone / MediaPipe Hand Detector
     │  (Identifies 21 landmarks & computes hand bounding box)
     ▼
Aspect-Ratio Preserving Normalization
     │  (Crops hand with 20px padding, resizes, and centers onto 300x300 white canvas)
     ▼
Universal Classifier (TFLite / Keras)
     │  (Predicts letter index 0-25 with confidence score in <15ms)
     ▼
Stability & Debounce Filter
     │  (Requires continuous prediction for 10 frames + cooldown before registering letter)
     ▼
Timing-Based Word Aggregator
     │  (Buffers letters; detects 2.5s pause to finalize word)
     ▼
Grammar & Autocorrect Processor
     │  (Cleans finger-spelling typos into readable English)
     ▼
Live Neubrutalism UI Overlay (OpenCV / Flask Web + Audio TTS)
```
