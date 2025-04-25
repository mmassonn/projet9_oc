import imgaug as ia
from imgaug import augmenters as iaa
import torch
from torch.nn import functional as F
from huggingface_hub import login, hf_hub_download
from huggingface_hub import HfApi
from flask import Flask, request, jsonify
from PIL import Image
from collections import namedtuple
import numpy as np
np.bool = np.bool_
np.complex = np.complex_
from matplotlib import colors
import base64
import io
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
HF_TOKEN = "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"
login(token=HF_TOKEN)

# Model
REPO_ID = "mmassonn/Segformer_tune"
MODEL_FILE_NAME = "Segformer_tune.pt"
model_file = hf_hub_download(repo_id=REPO_ID,filename=MODEL_FILE_NAME)
MODEL = torch.load(model_file, map_location="cpu", weights_only=False)

# Dataset
DATASET_IMAGE_REPO_ID = "mmassonn/CarSegmentation_leftImg8bit"
DATASET_MASK_REPO_ID = "mmassonn/CarSegmentation_gtFine"

def get_dataset_file_path() -> tuple[list, dict]:
    """Extrais les chemins d'accès aux fichier dans dataset."""
    dataset_file_paths = {}
    api = HfApi()

    # Use list_repo_files instead of list_models_files
    image_files = api.list_repo_files(repo_id=DATASET_IMAGE_REPO_ID, repo_type="dataset")
    image_dataset_file_names = [file for file in image_files]

    mask_files = api.list_repo_files(repo_id=DATASET_MASK_REPO_ID, repo_type="dataset")
    mask_dataset_file_names = [file for file in mask_files]

    # Pair image and mask files
    for image_file, mask_file in zip(image_dataset_file_names, mask_dataset_file_names):
        dataset_file_paths[image_file] = mask_file

    return image_dataset_file_names, dataset_file_paths

MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 512

def load_augmentation_aug_geometric() -> iaa.OneOf:
    """Applique des modifications géométriques des données dans le cadre de la data augmentation."""
    return iaa.OneOf([
        iaa.Sequential([iaa.Fliplr(0.2), iaa.Flipud(0)]),
        iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode='constant', pad_cval=(0, 255)),
        iaa.Crop(percent=(0.0, 0.3)),
        iaa.Crop(percent=(0.1, 0.5)),
        iaa.Crop(percent=(0.2, 0.4)),
        iaa.Crop(percent=(0.0, 0.25)),
        iaa.Sequential([
        iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                mode='constant',
                cval=(0, 255),
            )])
    ])

def load_augmentation_aug_non_geometric() -> iaa.OneOf:
    """Applique des modifications non géométriques des données dans le cadre de la data augmentation."""
    return iaa.OneOf([
        iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.2, iaa.MotionBlur(k=15, angle=[-45, 45])),
        iaa.Sometimes(0.2, iaa.MultiplyHue((0.5, 1.5))),
        iaa.Sometimes(0.34, iaa.Grayscale(alpha=(0.0, 1.0))),
        iaa.Sometimes(0.1, iaa.HistogramEqualization()),
        iaa.Sometimes(0.2, iaa.Rain(speed=(0.1, 0.2))),
        iaa.Sometimes(0.2, iaa.Fog()),
        iaa.Sometimes(0.2, iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)))
    ])

def load_augmentation_aug_all(test: bool = False) -> iaa.Sequential:
    """Applique selon une probabilité des modifications géométriques et non géométriques des données dans le cadre de la data augmentation."""
    if test:
        return iaa.Sequential([
        iaa.Sometimes(0.9, load_augmentation_aug_non_geometric()),
        iaa.Sometimes(0.9, load_augmentation_aug_geometric())
    ])
    else:
        return iaa.Sequential([
            iaa.Sometimes(0.5, load_augmentation_aug_non_geometric()),
            iaa.Sometimes(0.5, load_augmentation_aug_geometric())
        ])

def apply_augmentation_from_array(img_array):
    """Applique la data augmentation sur l'image et le masque."""
    augmentation_function = load_augmentation_aug_all(test=True)
    aug_det = augmentation_function.to_deterministic()
    return aug_det.augment_image(img_array)

def get_numpy_mask_from_image(mask_img):
    """Retourne un np.array avec 1 canaux, chacun représentant une caractéristique distincte."""
    conditions = [
    (mask_img >= 0) & (mask_img <= 6),
    (mask_img >= 7) & (mask_img <= 10),
    (mask_img >= 11) & (mask_img <= 16),
    (mask_img >= 17) & (mask_img <= 20),
    (mask_img >= 21) & (mask_img <= 22),
    (mask_img == 23),
    (mask_img >= 24) & (mask_img <= 25),
    ((mask_img >= 26) & (mask_img <= 33)) | (mask_img == -1)
    ]
    values = [0, 1, 2, 3, 4, 5, 6, 7]
    
    return np.select(conditions, values, default=mask_img)

def predict_segmentation(
    image_path: str, 
    image_width: int, 
    image_height: int,
    ):
    '''Genère le masque de couleur à partir du modèle.'''
    image = Image.open(image_path)
    image_array = np.array(image)
    resized_image = image.resize((image_width, image_height))
    resized_array = np.array(resized_image)
    img_tensor = (torch.from_numpy(resized_array).float().permute(2, 0, 1) / 255.0).unsqueeze(0)
    with torch.no_grad():
        outputs = MODEL(pixel_values=img_tensor.float())
        logits = outputs.logits
        upsampled_logits = F.interpolate(
            logits,
            size=(image_width, image_height),
            mode="bilinear",
            align_corners=False,
            )
    return upsampled_logits.argmax(dim=1)


app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcome on the segmentation API"

@app.route("/image_path", methods=["POST"])
def get_image_file_path():
    image_dataset_file_names, _ = get_dataset_file_path()
    return jsonify(image_dataset_file_names[1:])

@app.route("/process_image", methods=['GET', 'POST'])
def process_image() -> list:
    """Réalise le prétraitement d'une image à partir de son path."""
    file = request.get_json(force=True)
    image_file_name = file['file_name']
    image_path = hf_hub_download(
        repo_id=DATASET_IMAGE_REPO_ID,
        filename=image_file_name,
        repo_type="dataset",
        )
    image = Image.open(image_path)
    image_array = np.array(image)
    process_img = apply_augmentation_from_array(img_array=image_array)
    # Convertir les arrays en images encodées en base64
    def array_to_base64(arr):
        if isinstance(arr, Image.Image):
            img = arr
        else:
            img = Image.fromarray(arr.astype('uint8'))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        'image': array_to_base64(image_array),
        'process_img': array_to_base64(process_img),
    }

@app.route("/predict_mask", methods=['GET', 'POST'])
def segment_image() -> list:
    """Réalise la segmentation d'une image à partir de son path."""
    file = request.get_json(force=True)
    image_file_name = file['file_name']
    image_path = hf_hub_download(
        repo_id=DATASET_IMAGE_REPO_ID,
        filename=image_file_name,
        repo_type="dataset",
        )
    image = Image.open(image_path)
    image_array = np.array(image)
    pred_mask_tensor = predict_segmentation(
        image_path=image_path,
        image_width=MODEL_INPUT_WIDTH,
        image_height=MODEL_INPUT_HEIGHT,
        )
    _ , dataset_file_paths = get_dataset_file_path()
    real_mask_file_name = dataset_file_paths[image_file_name]
    real_mask_path = hf_hub_download(
        repo_id=DATASET_MASK_REPO_ID,
        filename=real_mask_file_name,
        repo_type="dataset",
        )
    pred_mask_array = pred_mask_tensor.squeeze(0).numpy()

    mask_array = np.array(Image.open(real_mask_path).convert('L'))
    mask_7_labels_array = get_numpy_mask_from_image(mask_img = mask_array) 
    # Convertir les arrays en images encodées en base64
    def array_to_base64(arr):
        if isinstance(arr, Image.Image):
            img = arr
        else:
            img = Image.fromarray(arr.astype('uint8'))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {
        'image': array_to_base64(image_array),
        'pred_mask': array_to_base64(pred_mask_array),
        'real_mask': array_to_base64(mask_7_labels_array),
    }
