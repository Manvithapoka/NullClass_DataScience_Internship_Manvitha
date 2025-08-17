# Task 3: Age and Emotion Detection from Voice

## 📌 Project Overview
This project is part of the NullClass Internship (Task 3) and aims to build a **machine learning model** that detects:
- **Age** from voice input  
- **Emotion** (e.g., happy, sad, angry, neutral, etc.) from voice  

The system can process recorded audio or real-time microphone input and predict **both** the speaker's **age group** and **emotional state**.

---

## 📂 Files in This Repository
- `model_training.ipynb` → Jupyter Notebook for model training  
- `voice_model.pkl` → Trained model file 
- `requirements.txt` → Required Python libraries  
- `gui_app.py` → Optional Streamlit/Tkinter GUI for predictions  
- `README.md` → Project documentation (this file)  

---

## 🚀 How It Works
1. **Audio Input** → User uploads a `.wav` or `.mp3` file or records live audio.  
2. **Feature Extraction** → Extracts MFCC and other audio features using **librosa**.  
3. **Model Prediction** → Predicts **age group** and **emotion** using trained models.  
4. **Output Display** → Shows the predicted results in the console or GUI.  

---

## 🛠️ Technologies Used
- **Python**
- **Librosa** (audio feature extraction)
- **NumPy & Pandas** (data handling)
- **Scikit-learn** (machine learning)
- **Streamlit** (optional GUI)
- **Joblib / Pickle** (model saving)

---
## 📂 DataSet

“The model was trained and evaluated using the RAVDESS/TESS dataset, which contains labeled audio recordings for emotion and age classification.”
