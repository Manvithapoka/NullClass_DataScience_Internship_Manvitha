import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import gradio as gr

# ===============================
# Load and preprocess dataset
# ===============================
df = pd.read_csv("labels.csv")

# Map categorical values
df['hair_length'] = df['hair_length'].map({'Short': 0, 'Long': 1})
df['gender_label'] = df['gender'].map({'Male': 0, 'Female': 1})

# Apply custom rule-based label
def apply_task_logic(row):
    if 20 <= row['age'] <= 30:
        return 1 if row['hair_length'] == 1 else 0  # Female if long hair
    else:
        return row['gender_label']

df['target'] = df.apply(apply_task_logic, axis=1)

# Features and target
X = df[['age', 'hair_length']]
y = df['target']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# ===============================
# Prediction function for GUI
# ===============================
def predict_gender_gui(age, hair_length):
    if 20 <= age <= 30:
        # Rule-based prediction
        return "Female" if hair_length == "Long" else "Male"
    else:
        # Model prediction
        hair = 1 if hair_length == "Long" else 0
        input_data = [[age, hair]]
        prediction = model.predict(input_data)[0]
        return "Female" if prediction == 1 else "Male"

# ===============================
# Gradio Interface
# ===============================
iface = gr.Interface(
    fn=predict_gender_gui,
    inputs=[
        gr.Slider(0, 100, step=1, label="Age"),
        gr.Radio(["Short", "Long"], label="Hair Length")
    ],
    outputs="text",
    title="Long Hair Identification",
    description="Predict gender based on age and hair length using custom rules."
)

if __name__ == "__main__":
    iface.launch()
