# 🎭 Mood Mirror: Emotion-Based Song Recommender


Mood Mirror is a Streamlit-based web app that uses facial emotion detection to recommend songs that match the user's mood. Just upload a selfie or face image, and let the app suggest music tailored to your emotional state!


🌟 Features
🎯 Emotion Detection from facial expressions using a CNN model (emotion_model.h5)

🎵 Music Recommendation based on mapped moods using a curated Spotify dataset (data_moods.csv)

🖼️ Simple and intuitive Streamlit interface

🔁 Real-time recommendation engine powered by your detected mood

🚀 How It Works
1. Upload a Face Image (JPG/PNG)

2. Detect Emotion using the pretrained Keras model

3. Map Emotion to Mood (e.g., Happy → Happy, Fear → Calm)

4. Recommend Songs from a Spotify dataset filtered by mood

Enjoy Your Personalized Playlist! 🎧

# Setup Instructions
1. Clone the Repository

git clone https://github.com/Devika499/mood-mirror.git

cd mood-mirror

2. Install Dependencies

pip install -r requirements.txt

requirements.txt should include: streamlit, keras, opencv-python, numpy, pillow, pandas

3. Run the App

streamlit run app.py

# 📊 Dataset Source

Spotify Mood Dataset: data_moods.csv contains song titles, artists, and mood labels.

Emotion detection uses a 48x48 grayscale facial image format based on FER-2013-like inputs.

# 🔐 Requirements

Python 3.7+

Keras / TensorFlow backend

OpenCV

Streamlit

# 🙌 Acknowledgements

Emotion detection model based on [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

Music data adapted from public Spotify mood datasets
