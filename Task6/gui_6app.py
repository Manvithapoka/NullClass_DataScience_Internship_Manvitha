import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
from deepface import DeepFace
import os

# Load model and mappings
try:
    model = joblib.load("nationality_model.pkl")
    race_mapping = joblib.load("race_mapping.pkl")
    nationality_mapping = joblib.load("nationality_mapping.pkl")
except FileNotFoundError:
    messagebox.showerror("Error", "Model or mapping files not found! Place .pkl files in the same folder.")
    exit()

# Reverse mappings for display
reverse_nat_mapping = {v: k for k, v in nationality_mapping.items()}
reverse_race_mapping = {v: k for k, v in race_mapping.items()}

# Prediction function
def predict_nationality(img_path):
    try:
        analysis = DeepFace.analyze(img_path, actions=['race'], enforce_detection=False)
        race = analysis[0]['dominant_race']
        race_id = race_mapping.get(race, -1)

        if race_id == -1:
            return f"Race '{race}' not recognized in training data."

        pred_nat_id = model.predict([[race_id]])[0]
        return reverse_nat_mapping[pred_nat_id]
    except Exception as e:
        return f"Error: {str(e)}"

# File upload handler
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    nationality = predict_nationality(file_path)
    result_label.config(text=f"Predicted Nationality: {nationality}")

# GUI setup
root = tk.Tk()
root.title("Nationality Detection")
root.geometry("400x200")
root.resizable(False, False)

title_label = tk.Label(root, text="Nationality Detection", font=("Arial", 16))
title_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Image", command=upload_image, font=("Arial", 12))
upload_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
