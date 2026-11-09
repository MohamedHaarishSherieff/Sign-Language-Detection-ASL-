import os
import sys

# Ensure local config directory for matplotlib to prevent Windows permission errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.mplconfig'))

import cv2
import time
from sign_detector import SignLanguageDetector

def main():
    print("=" * 65)
    print("  AMERICAN SIGN LANGUAGE (ASL) REAL-TIME TRANSLATOR")
    print("  Desktop OpenCV Mode")
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
        print("ERROR: Could not open webcam. Ensure your camera is connected and not in use by another app.")
        return

    # Set camera resolution
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
                print("Space added. Current translation:", detector.get_translation())
            elif key == 8 or key == ord('b'):  # Backspace or 'b'
                detector.backspace()
                print("Backspace. Current translation:", detector.get_translation())
            elif key == ord('c'):              # Clear
                detector.reset()
                print("Translation cleared.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("\nApplication closed cleanly.")

if __name__ == "__main__":
    main()

