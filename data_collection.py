import os
import sys

# Ensure local config directory for matplotlib to prevent Windows permission errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(BASE_DIR, '.mplconfig'))

import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

def main():
    print("=" * 65)
    print("  ASL DATA COLLECTION TOOL")
    print("=" * 65)
    letter = input("Enter the sign/letter you want to collect (A-Z): ").strip().upper()
    if not letter:
        letter = "A"

    folder = os.path.join(BASE_DIR, "Data", letter)
    os.makedirs(folder, exist_ok=True)

    existing_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    counter = len(existing_files)
    print(f"Target folder: {folder}")
    print(f"Already existing images for '{letter}': {counter}")
    print("\nControls:")
    print("  - Press 's' to capture and save an image")
    print("  - Press 'q' or [Esc] to exit")
    print("=" * 65)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector = HandDetector(maxHands=1)
    offset = 20
    imgSize = 300

    try:
        while True:
            success, img = cap.read()
            if not success:
                continue

            h_img, w_img, _ = img.shape
            hands, imgWithHands = detector.findHands(img, draw=True)

            imgWhite = None
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Clamp coordinates safely
                y1 = max(0, y - offset)
                y2 = min(h_img, y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(w_img, x + w + offset)

                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    aspectRatio = h / max(w, 1)

                    if aspectRatio > 1:
                        k = imgSize / max(h, 1)
                        wCal = max(1, math.ceil(k * w))
                        imgResize = cv2.resize(imgCrop, (min(wCal, imgSize), imgSize))
                        wGap = math.ceil((imgSize - imgResize.shape[1]) / 2)
                        imgWhite[:, wGap:imgResize.shape[1] + wGap] = imgResize
                    else:
                        k = imgSize / max(w, 1)
                        hCal = max(1, math.ceil(k * h))
                        imgResize = cv2.resize(imgCrop, (imgSize, min(hCal, imgSize)))
                        hGap = math.ceil((imgSize - imgResize.shape[0]) / 2)
                        imgWhite[hGap:imgResize.shape[0] + hGap, :] = imgResize

                    cv2.imshow("Hand Crop (Processed)", imgWhite)

            # Neubrutalism HUD at the top
            hud_h = 70
            cv2.rectangle(img, (0, 0), (w_img, hud_h), (10, 10, 10), cv2.FILLED)
            cv2.rectangle(img, (0, hud_h - 4), (w_img, hud_h), (0, 230, 255), cv2.FILLED)

            cv2.putText(img, f"COLLECTING: [{letter}]  //  SAVED: {counter} IMAGES", (20, 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 230, 255), 2)
            cv2.putText(img, "CONTROLS: PRESS 'S' TO CAPTURE  //  'Q' OR [ESC] TO QUIT", (20, 58),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Data Collector", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and imgWhite is not None:
                counter += 1
                filename = os.path.join(folder, f"Image_{int(time.time() * 1000)}.jpg")
                cv2.imwrite(filename, imgWhite)
                print(f"[+] Saved image #{counter} to {filename}")
            elif key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinished session for '{letter}'. Total images: {counter}")

if __name__ == "__main__":
    main()

