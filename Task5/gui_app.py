import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import joblib  # for loading .pkl model
import numpy as np

class CarColourDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Colour Detection")
        self.root.geometry("500x500")

        # Load your trained model here
        try:
            self.model = joblib.load("savedmodel.pkl")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.model = None

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.btn_load = tk.Button(root, text="Load Car Image", command=self.load_image)
        self.btn_load.pack(pady=5)

        self.btn_predict = tk.Button(root, text="Detect Colour", command=self.detect_colour)
        self.btn_predict.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

        self.img_path = None
        self.loaded_image = None

    def load_image(self):
        filetypes = [("Image Files", "*.jpg *.jpeg *.png")]
        self.img_path = filedialog.askopenfilename(title="Choose a car image", filetypes=filetypes)
        if self.img_path:
            img = Image.open(self.img_path)
            img = img.resize((300, 200))
            self.loaded_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.loaded_image)
            self.result_label.config(text="")

    def detect_colour(self):
        if not self.img_path:
            messagebox.showwarning("No Image", "Please load a car image first.")
            return

        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded properly.")
            return

        # Preprocess the image as your model expects
        # For example, convert image to numpy array, resize, flatten, normalize, etc.
        try:
            img = Image.open(self.img_path)
            img = img.resize((64, 64))  # example size, change as per your model
            img_np = np.array(img)
            img_np = img_np.flatten().reshape(1, -1)  # flatten if model expects 1D input

            # Predict using your model
            prediction = self.model.predict(img_np)
            detected_color = prediction[0]  # adjust if your model output differs

            self.result_label.config(text=f"Detected Colour: {detected_color}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CarColourDetectorApp(root)
    root.mainloop()
