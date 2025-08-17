# Task 2: Senior Citizen Identification

## 🔍 Description
This project detects multiple persons from a video or real-time webcam feed, identifies their age and gender, and determines if they are senior citizens (age 60+). It logs the **age, gender, and time of visit** to a CSV file.

---

## ✅ Objectives

- Detect multiple faces in a video/webcam stream
- Predict **age** and **gender**
- Identify if a person is a **senior citizen (age > 60)**
- Log results in a CSV file (`log.csv`) with:
  - Age
  - Gender
  - Time of detection

---

## 🛠️ Tools & Libraries Used

- Python
- OpenCV
- DeepFace
- Pandas

---

## 📁 Files in this folder

- `Senior_Citizen_Identification.ipynb` – Colab notebook with full implementation
- `requirements.txt` – Python packages needed
- `log.csv` – Output log of detections

---

## 📌 Notes

- The model works on webcam or a video input.
- GUI is optional; this version focuses on backend functionality.

## Dataset

This task did not use a custom dataset.  

Instead, a pre-trained age and gender detection model was utilized for inference.  
The model internally uses large-scale datasets (such as UTKFace) for training, but since it was pre-trained, no separate dataset was added here.  
