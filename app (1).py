import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Charger le modèle entraîné
model = tf.keras.models.load_model('/content/drive/MyDrive/PFA/chest_xray_model.h5')

# Fonction pour prédire une image
def predict(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    prediction = model.predict(img_array)
    return prediction[0][0]

# Interface utilisateur avec Streamlit
st.markdown("<h1 style='text-align: center;'>LungCheck</h1>", unsafe_allow_html=True)
st.markdown("<h3> A propos : </h3>", unsafe_allow_html=True)
st.markdown("LungCheck est un site web alimenté par l'intelligence artificielle, conçue pour permettre aux utilisateurs de télécharger des radiographies thoraciques et de recevoir une analyse automatisée pour détecter la pneumonie. Elle offre une interface conviviale, garantit la sécurité des données médicales et utilise des algorithmes de deep learning pour fournir des résultats précis et fiables.")
st.markdown("<h3> Téléchargez une image : </h3> ", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image Téléchargée', use_column_width=True)
    
    # Sauvegarder l'image pour la prédiction
    image_path = 'temp_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Prédiction
    st.write("Classification en cours...")
    prediction = predict(image_path, model)
    
    if prediction > 0.5:
        st.markdown("<p style='color: red; font-size: 24px;'>Prédiction : Pneumonie</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: green; font-size: 24px;'>Prédiction : Normal</p>", unsafe_allow_html=True)
