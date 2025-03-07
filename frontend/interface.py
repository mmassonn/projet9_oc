import streamlit as st
import requests
import numpy as np
from PIL import Image
import base64
import io

DATA_API_URL = "https://carsegmentationwebapp-bgggfnfgefdchrgp.francecentral-01.azurewebsites.net/image_path"
PREDICTION_API_URL = "https://carsegmentationwebapp-bgggfnfgefdchrgp.francecentral-01.azurewebsites.net/predict_mask"

def get_file_list_from_api():
    """Récupére de la liste des images disponibles depuis l'api."""
    data_api_url = DATA_API_URL
    response = requests.post(data_api_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération de la liste des fichiers.")
        return []

def send_post_request(selected_file):
    """Envoie à l'api le nom de l'image sélectionnée."""
    url = PREDICTION_API_URL
    files = {'file_name': str(selected_file)}
    response = requests.post(url, json=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de l'envoi de la requête POST.")
        return None

def display_images(images, target_size=(300, 300)):
    """Affiche l'image réelle, le masque segmentée réelle et celui réalisé par le modèle."""
    descriptions = {
        'image': "Image réelle",
        'pred_mask': "Segmentation réalisée par le modèle",
        'real_mask': "Segmentation réelle",
    }

    cols = st.columns(len(images))
    
    for idx, (key, base64_string) in enumerate(images.items()):
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img_resized = img.resize(target_size, Image.LANCZOS)
        array = np.array(img_resized)
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        with cols[idx]:
            st.image(array, caption=descriptions[key], use_container_width=True)

st.title("Application de Segmentation d'Image")

file_list = get_file_list_from_api()
selected_file = st.selectbox("Sélectionnez un fichier", file_list)

if st.button("Création de l'image segmentée"):
    if selected_file:
        images = send_post_request(selected_file)
        if images:
            display_images(images)
        else:
            st.error("Erreur lors de la récupération du fichier sélectionné.")
    else:
        st.warning("Veuillez sélectionner un fichier.")
