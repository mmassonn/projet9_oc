import streamlit as st
import requests
import numpy as np
from PIL import Image
import base64
import io

DATA_API_URL = "http://51.20.98.229:5000/image_path"
PROCESSING_API_URL = "http://51.20.98.229:5000/process_image"
PREDICTION_API_URL = "http://51.20.98.229:5000/predict_mask"

def get_file_list_from_api():
    """Récupére de la liste des images disponibles depuis l'api."""
    data_api_url = DATA_API_URL
    response = requests.post(data_api_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de la récupération de la liste des fichiers.")
        return []

def send_post_request_processing(selected_file):
    """Envoie à l'api pour le prétraitement, le nom de l'image sélectionnée."""
    url = PROCESSING_API_URL
    files = {'file_name': str(selected_file)}
    response = requests.post(url, json=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de l'envoi de la requête POST.")
        return None

def send_post_request_prediction(selected_file):
    """Envoie à l'api pour la prédiction le nom de l'image sélectionnée."""
    url = PREDICTION_API_URL
    files = {'file_name': str(selected_file)}
    response = requests.post(url, json=files)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Erreur lors de l'envoi de la requête POST.")
        return None


# Définir votre palette de couleurs personnalisée (7 couleurs max)
colors_palette = [
    [0, 0, 255],    # Bleu (b)
    [0, 255, 0],    # Vert (g)
    [255, 0, 0],    # Rouge (r)
    [0, 255, 255],  # Cyan (c)
    [255, 0, 255],  # Magenta (m)
    [255, 255, 0],  # Jaune (y)
    [110, 50, 30],  # Marron (y)
    [0, 0, 0]       # Noir (k)
]
colors_array = np.array(colors_palette, dtype=np.uint8)

def apply_custom_cmap(array):
    """Applique la palette de couleurs à un array 2D"""
    normalized = (array - array.min()) * (len(colors_array)-1) / (array.max() - array.min() + 1e-10)
    indices = np.clip(normalized.astype(int), 0, len(colors_array)-1)
    return colors_array[indices]

def display_process_images(images, target_size=(512, 512)):
    """Affiche l'image réelle, le masque segmentée réelle et celui réalisé par le modèle."""
    descriptions = {
        'image': "Image avant transformation",
        'process_img': "Image après transformation",
    }

    cols = st.columns(len(images))    
    for idx, (key, base64_string) in enumerate(images.items()):
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img_resized = img.resize(target_size, Image.LANCZOS)
        array = np.array(img_resized)
        # Conversion de type et application de la palette
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        with cols[idx]:
            st.image(array, caption=descriptions[key], use_container_width=True)

def display_images(images, target_size=(512, 512)):
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
        # Conversion de type et application de la palette
        if array.ndim == 2:  # Si image grayscale
            if array.dtype != np.uint8:
                array = (array * 255).astype(np.uint8)
            array = apply_custom_cmap(array)
        elif array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        with cols[idx]:
            st.image(array, caption=descriptions[key], use_container_width=True)

# Repère principal (landmark) pour les lecteurs d'écran
st.markdown('<main id="main-content" tabindex="-1"></main>', unsafe_allow_html=True)

# Titre principal unique (équivalent <h1>)
st.title("Segmentation d'une scène urbaine")

file_list = get_file_list_from_api()

col1, col2 = st.columns(2)

# Partie gauche : Prétraitement
with col1:
    st.header("Prétraitement d'une image")  # <h2>
    selected_file_pret = st.selectbox(
        "Sélectionnez un fichier à prétraiter",
        file_list,
        key="pret"
    )
    if st.button("Réaliser le prétraitement de l'image"):
        if selected_file_pret:
            images_pret = send_post_request_processing(selected_file_pret)
            if images_pret:
                display_process_images(images_pret)
            else:
                st.error("Erreur lors du prétraitement de l'image sélectionnée.")
        else:
            st.warning("Veuillez sélectionner un fichier.")

# Partie droite : Prédiction
with col2:
    st.header("Réaliser la prédiction")  # <h2>
    selected_file_pred = st.selectbox(
        "Sélectionnez un fichier à prédire",
        file_list,
        key="pred"
    )
    if st.button("Création de l'image segmentée"):
        if selected_file_pred:
            images_pred = send_post_request_prediction(selected_file_pred)
            if images_pred:
                display_images(images_pred)
            else:
                st.error("Erreur lors de la récupération du fichier sélectionné.")
        else:
            st.warning("Veuillez sélectionner un fichier.")
