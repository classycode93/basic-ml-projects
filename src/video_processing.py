
import cv2
import os

def extract_frames(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    i = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(output_folder, f"{i}.jpg")

        cv2.imwrite(frame_path, frame)

        i += 1

    cap.release()

    print(f"Extracted {i} frames from {video_path}")
