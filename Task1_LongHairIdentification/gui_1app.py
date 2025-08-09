import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

class LongHairApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Long Hair Identification")
        self.root.geometry("500x550")

        try:
            self.model = tf.keras.models.load_model("longhair_cnn_model.h5")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.model = None

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack(pady=5)

        self.btn_predict = tk.Button(root, text="Predict Long Hair", command=self.predict_long_hair)
        self.btn_predict.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

        self.img_path = None
        self.loaded_image = None

    def load_image(self):
        filetypes = [("Image Files", "*.jpg *.jpeg *.png")]
        self.img_path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
        if self.img_path:
            img = Image.open(self.img_path)
            img = img.resize((400, 300))
            self.loaded_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.loaded_image)
            self.result_label.config(text="")

    def predict_long_hair(self):
        if not self.img_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded properly.")
            return

        try:
            img = Image.open(self.img_path).resize((64,64))
            img_array = np.array(img)/255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            prediction = self.model.predict(img_array)[0][0]
            result_text = "Long Hair" if prediction > 0.5 else "No Long Hair"

            self.result_label.config(text=f"Prediction: {result_text}")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LongHairApp(root)
    root.mainloop()
