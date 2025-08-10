import gradio as gr
from deepface import DeepFace
import cv2
from datetime import datetime
import pandas as pd
import os

# ===============================
# Function for video processing
# ===============================
def detect_seniors(video_file):
    # Load video
    cap = cv2.VideoCapture(video_file.name)

    senior_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = DeepFace.analyze(frame, actions=['age', 'gender'], enforce_detection=False)

            if not isinstance(results, list):
                results = [results]

            for person in results:
                age = person['age']
                gender = person['gender']

                print(f"Detected person --> Age: {age} | Gender: {gender}")

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                senior_data.append([int(age), gender, now])

        except Exception as e:
            print("Error analyzing frame:", e)

    cap.release()

    # Save detections to CSV
    df = pd.DataFrame(senior_data, columns=["Age", "Gender", "Time"])
    output_csv = "senior_visits.csv"
    df.to_csv(output_csv, index=False)

    return output_csv  # Return CSV file path

# ===============================
# Gradio Interface
# ===============================
iface = gr.Interface(
    fn=detect_seniors,
    inputs=gr.File(file_types=[".mp4", ".avi", ".mov"], label="Upload Video"),
    outputs=gr.File(label="Download Senior Visits CSV"),
    title="Senior Citizen Detection",
    description="Upload a video to detect age and gender of people. Saves results to CSV."
)

if __name__ == "__main__":
    iface.launch()
