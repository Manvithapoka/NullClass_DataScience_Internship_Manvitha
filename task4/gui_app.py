import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import pickle
import datetime

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Time restriction (6 PM to 10 PM)
def is_within_time():
    current_hour = datetime.datetime.now().hour
    return 18 <= current_hour < 22

# Predict from image
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    label = label_encoder.inverse_transform([predicted_class])[0]
    return label

# GUI Setup
def upload_image():
    if not is_within_time():
        result_label.config(text="This app works only from 6 PM to 10 PM")
        return
    
    file_path = filedialog.askopenfilename()
    if file_path:
        label = predict_image(file_path)
        result_label.config(text=f"Prediction: {label}")

        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

# Initialize GUI
root = tk.Tk()
root.title("Sign Language Detection")
root.geometry("400x400")

upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="Upload an image to predict", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
