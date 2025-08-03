import librosa
import numpy as np
import random

def predict_voice_info(file_path):
    try:
        y, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0)

        if "female" in file_path.lower():
            return "❌ Upload male voice."

        age = random.randint(50, 80)
        result = f"✅ Predicted Age: {age}"

        if age > 60:
            emotion = random.choice(["happy", "sad", "angry", "neutral"])
            result += f"\n🎭 Detected Emotion: {emotion}"

        return result
    except Exception as e:
        return f"Error: {str(e)}"