import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import pandas as pd
import tempfile
from PIL import Image

# Load the pre-trained emotion detection model
emotion_model = load_model('emotion_model.h5')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load Spotify mood dataset
songs_df = pd.read_csv('data_moods.csv')

# Emotion to mood mapping
emotion_to_mood = {
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'energetic',
    'surprise': 'energetic',
    'neutral': 'calm',
    'fear': 'calm',
    'disgust': 'calm'
}

# Emotion detection function
def detect_emotion_from_image(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)
    
    predictions = emotion_model.predict(face)
    emotion_index = np.argmax(predictions)
    detected_emotion = emotion_labels[emotion_index]
    return detected_emotion

# Recommend songs based on emotion
def recommend_songs(emotion, top_n=5):
    mood = emotion_to_mood.get(emotion, 'calm')
    filtered_songs = songs_df[songs_df['mood'].str.lower() == mood]
    
    if filtered_songs.empty:
        return [f"No songs found for mood: {mood}"]
    
    filtered_songs = filtered_songs.sample(top_n)
    return [f"{row['name']} by {row['artist']}" for _, row in filtered_songs.iterrows()]

# Streamlit UI
st.set_page_config(page_title="Mood Mirror", page_icon="🎵")
st.title("🎭 Mood Mirror: Emotion-Based Song Recommender")

uploaded_file = st.file_uploader("Upload an image of a face", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Emotion and Recommend Songs"):
        try:
            emotion = detect_emotion_from_image(img)
            st.success(f"Detected Emotion: **{emotion.capitalize()}**")

            recommendations = recommend_songs(emotion)
            st.subheader("🎧 Recommended Songs:")
            for song in recommendations:
                st.write(f"- {song}")
        except Exception as e:
            st.error(f"Error: {e}")
