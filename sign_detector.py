import os
import sys

# Silence TensorFlow verbose logs and oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure local config directory for matplotlib to prevent Windows permission errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.mplconfig'))

import cv2
import numpy as np
import math
import time
from threading import Lock
from cvzone.HandTrackingModule import HandDetector
from grammar_processor import GrammarProcessor


class UniversalClassifier:
    """
    Universal classifier that supports both TFLite (recommended for speed and cross-version stability)
    and legacy Keras H5 models.
    """
    def __init__(self, model_path=None, labels_path=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        if model_path is None:
            model_path = os.path.join(self.base_dir, "Model", "model.tflite")
            if not os.path.exists(model_path):
                model_path = os.path.join(self.base_dir, "Model", "keras_model.h5")

        if labels_path is None:
            labels_path = os.path.join(self.base_dir, "Model", "labels.txt")

        # Load labels
        self.labels = []
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.labels.append(parts[1])
                    elif len(parts) == 1:
                        self.labels.append(parts[0])
        if not self.labels:
            self.labels = [chr(65 + i) for i in range(26)]

        # Try TFLite interpreter first (works seamlessly on Keras 2, Keras 3, and TF 2.x)
        self.interpreter = None
        tflite_candidate = model_path if model_path.endswith('.tflite') else os.path.join(self.base_dir, "Model", "model.tflite")

        if os.path.exists(tflite_candidate):
            try:
                import tensorflow as tf
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.interpreter = tf.lite.Interpreter(model_path=tflite_candidate)
                    self.interpreter.allocate_tensors()
                    self.input_details = self.interpreter.get_input_details()
                    self.output_details = self.interpreter.get_output_details()
            except Exception:
                self.interpreter = None

        # Fallback to Keras / cvzone
        self.cvzone_classifier = None
        self.keras_model = None
        if self.interpreter is None:
            h5_path = os.path.join(self.base_dir, "Model", "keras_model.h5")
            try:
                from cvzone.ClassificationModule import Classifier
                self.cvzone_classifier = Classifier(h5_path, labels_path)
            except Exception:
                try:
                    import tensorflow as tf
                    self.keras_model = tf.keras.models.load_model(h5_path, compile=False)
                except Exception as e:
                    print(f"Warning: Could not initialize Keras fallback: {e}")

    def getPrediction(self, img, draw=False):
        if self.interpreter is not None:
            # Resize and normalize for Teachable Machine MobileNet
            img_resized = cv2.resize(img, (224, 224))
            normalized = (img_resized.astype(np.float32) / 127.0) - 1.0
            input_tensor = np.expand_dims(normalized, axis=0)

            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            index = int(np.argmax(prediction))
            return list(prediction), index

        elif self.cvzone_classifier is not None:
            return self.cvzone_classifier.getPrediction(img, draw=draw)

        elif self.keras_model is not None:
            img_resized = cv2.resize(img, (224, 224))
            normalized = (img_resized.astype(np.float32) / 127.0) - 1.0
            input_tensor = np.expand_dims(normalized, axis=0)
            prediction = self.keras_model.predict(input_tensor, verbose=0)[0]
            index = int(np.argmax(prediction))
            return list(prediction), index

        return [0.0] * len(self.labels), 0


class SignLanguageDetector:
    def __init__(self, model_path=None, labels_path=None, camera_index=0):
        self.base_dir = BASE_DIR

        # Initialize detector & universal classifier
        self.detector = HandDetector(maxHands=1, detectionCon=0.7)
        self.classifier = UniversalClassifier(model_path, labels_path)
        self.labels = self.classifier.labels
        self.grammar_processor = GrammarProcessor()

        # Camera & image processing settings
        self.camera_index = camera_index
        self.cap = None
        self._init_camera()

        self.offset = 20
        self.imgSize = 300

        # State tracking
        self.lock = Lock()
        self.current_letter = ""
        self.current_confidence = 0.0

        # Debounce / stabilization parameters
        self.candidate_letter = ""
        self.stable_frame_count = 0
        self.required_stable_frames = 10  # ~0.35s steady hand hold
        self.cooldown_seconds = 0.8       # Minimum time between registered letters
        self.last_letter_time = 0
        self.last_detection_time = time.time()
        self.word_pause_threshold = 2.5   # Seconds of inactivity before finalizing word

        # Text buffers
        self.current_word = []
        self.full_sentence = []
        self.display_translation = ""

    def _init_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # Try DirectShow on Windows for faster camera startup
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

    def process_frame(self, img):
        if img is None:
            return None

        h_img, w_img, _ = img.shape
        imgOutput = img.copy()
        current_time = time.time()

        # Find hands
        hands, _ = self.detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Safely clamp coordinates to avoid OpenCV out-of-bounds errors
            y1 = max(0, y - self.offset)
            y2 = min(h_img, y + h + self.offset)
            x1 = max(0, x - self.offset)
            x2 = min(w_img, x + w + self.offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                aspectRatio = h / max(w, 1)

                try:
                    if aspectRatio > 1:
                        k = self.imgSize / max(h, 1)
                        wCal = max(1, math.ceil(k * w))
                        imgResize = cv2.resize(imgCrop, (min(wCal, self.imgSize), self.imgSize))
                        wGap = math.ceil((self.imgSize - imgResize.shape[1]) / 2)
                        imgWhite[:, wGap:imgResize.shape[1] + wGap] = imgResize
                    else:
                        k = self.imgSize / max(w, 1)
                        hCal = max(1, math.ceil(k * h))
                        imgResize = cv2.resize(imgCrop, (self.imgSize, min(hCal, self.imgSize)))
                        hGap = math.ceil((self.imgSize - imgResize.shape[0]) / 2)
                        imgWhite[hGap:imgResize.shape[0] + hGap, :] = imgResize

                    prediction, index = self.classifier.getPrediction(imgWhite, draw=False)
                    confidence = float(np.max(prediction))

                    if 0 <= index < len(self.labels):
                        detected_letter = self.labels[index]
                        self.current_letter = detected_letter
                        self.current_confidence = confidence

                        # Only process high-confidence detections
                        if confidence >= 0.65:
                            if detected_letter == self.candidate_letter:
                                self.stable_frame_count += 1
                            else:
                                self.candidate_letter = detected_letter
                                self.stable_frame_count = 1

                            # Check if the sign has been held steadily
                            if self.stable_frame_count >= self.required_stable_frames:
                                if current_time - self.last_letter_time >= self.cooldown_seconds:
                                    with self.lock:
                                        self.current_word.append(detected_letter)
                                        self.last_letter_time = current_time
                                        self.stable_frame_count = 0
                                        self._update_display_translation()

                        self.last_detection_time = current_time

                        # Visual: Neubrutalism High-Contrast Overlay
                        progress = min(1.0, self.stable_frame_count / self.required_stable_frames)
                        # Neubrutalism Colors: High-Voltage Yellow (0, 230, 255) tracking, Neon Mint Green (50, 255, 50) locked
                        theme_color = (50, 255, 50) if progress >= 1.0 else (0, 230, 255)

                        # Draw thick black shadow/outline first for hard Neubrutalist contrast
                        cv2.rectangle(imgOutput, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 5)
                        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), theme_color, 3)

                        # Top label sticker badge (Solid yellow/green box with 2px black border and bold black text)
                        label_text = f" {detected_letter}  {int(confidence * 100)}% "
                        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
                        banner_y1 = max(0, y1 - 38)
                        banner_w = max(tw + 10, 110)
                        cv2.rectangle(imgOutput, (x1, banner_y1), (x1 + banner_w, y1), (0, 0, 0), cv2.FILLED)
                        cv2.rectangle(imgOutput, (x1 + 2, banner_y1 + 2), (x1 + banner_w - 2, y1 - 2), theme_color, cv2.FILLED)
                        cv2.putText(imgOutput, label_text, (x1 + 4, y1 - 10),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)

                        # Stability progress bar below hand (Black container with bright fill)
                        bar_w = x2 - x1
                        bar_y = min(h_img - 14, y2 + 10)
                        fill_w = int(bar_w * progress)
                        cv2.rectangle(imgOutput, (x1, bar_y), (x2, bar_y + 10), (0, 0, 0), cv2.FILLED)
                        cv2.rectangle(imgOutput, (x1, bar_y), (x2, bar_y + 10), (255, 255, 255), 1)
                        if fill_w > 0:
                            cv2.rectangle(imgOutput, (x1 + 1, bar_y + 1), (x1 + fill_w - 1, bar_y + 9), (50, 255, 50), cv2.FILLED)

                except Exception:
                    pass
        else:
            # Check for word completion on inactivity pause
            if self.current_word and (current_time - self.last_detection_time > self.word_pause_threshold):
                with self.lock:
                    self._finalize_current_word()

        # Draw HUD bar at the bottom
        imgOutput = self._draw_hud(imgOutput)
        return imgOutput

    def _draw_hud(self, img):
        h, w, _ = img.shape
        hud_h = 75
        # High-contrast Neubrutalist HUD with a thick 4px yellow accent border
        cv2.rectangle(img, (0, h - hud_h), (w, h), (10, 10, 10), cv2.FILLED)
        cv2.rectangle(img, (0, h - hud_h), (w, h - hud_h + 4), (0, 230, 255), cv2.FILLED)

        with self.lock:
            current_building = "".join(self.current_word).upper()
            sentence_str = " ".join(self.full_sentence).upper()
            if current_building:
                text_to_show = f"SENTENCE: {sentence_str} // [{current_building}]" if sentence_str else f"BUILDING: [{current_building}]"
            else:
                text_to_show = f"SENTENCE: {sentence_str}" if sentence_str else "STATUS: WAITING FOR SIGNS... (HOLD SIGN STEADILY)"

        cv2.putText(img, text_to_show, (20, h - 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        return img

    def _finalize_current_word(self):
        if not self.current_word:
            return
        raw_word = "".join(self.current_word)
        corrected_word = self.grammar_processor.correct_sentence(raw_word)
        if not corrected_word:
            corrected_word = raw_word
        self.full_sentence.append(corrected_word)
        self.current_word = []
        self._update_display_translation()

    def _update_display_translation(self):
        current_building = "".join(self.current_word)
        words = list(self.full_sentence)
        if current_building:
            words.append(current_building)
        self.display_translation = " ".join(words)

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self._init_camera()
            if not self.cap.isOpened():
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Webcam not available", (150, 240),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()

        success, frame = self.cap.read()
        if not success:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Error reading webcam frame", (120, 240),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            return buffer.tobytes()

        processed = self.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', processed)
        return buffer.tobytes()

    def add_space(self):
        with self.lock:
            self._finalize_current_word()
            self._update_display_translation()

    def backspace(self):
        with self.lock:
            if self.current_word:
                self.current_word.pop()
            elif self.full_sentence:
                self.full_sentence.pop()
            self._update_display_translation()

    def reset(self):
        with self.lock:
            self.current_word = []
            self.full_sentence = []
            self.display_translation = ""
            self.candidate_letter = ""
            self.stable_frame_count = 0
            self.last_detection_time = time.time()

    def get_translation(self):
        with self.lock:
            return self.display_translation

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __del__(self):
        self.release()


# Direct execution entry point
if __name__ == "__main__":
    print("=" * 65)
    print("  AMERICAN SIGN LANGUAGE (ASL) REAL-TIME TRANSLATOR")
    print("  Neubrutalism Live Camera Edition")
    print("=" * 65)
    print("Controls:")
    print("  - Hold a sign steadily to register a letter (green progress bar)")
    print("  - [Space]: Add space / complete word")
    print("  - [Backspace] or 'b': Delete last character")
    print("  - 'c': Clear / reset translation")
    print("  - 'q' or [Esc]: Exit")
    print("=" * 65)

    detector = SignLanguageDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("\n[!] ERROR: Could not access webcam. Please check if your camera is plugged in and not in use by another application.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "ASL Real-Time Translator // Neubrutalism Edition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            output_frame = detector.process_frame(frame)
            if output_frame is None:
                output_frame = frame

            cv2.imshow(window_name, output_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:   # 'q' or Esc
                break
            elif key == ord(' '):              # Space
                detector.add_space()
                print("[+] Space added. Current text:", detector.get_translation())
            elif key == 8 or key == ord('b'):  # Backspace or 'b'
                detector.backspace()
                print("[+] Backspace. Current text:", detector.get_translation())
            elif key == ord('c'):              # Clear
                detector.reset()
                print("[+] Translation cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("\nApplication closed successfully.")
