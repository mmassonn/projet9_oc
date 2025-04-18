import torch
from torch.nn import functional as F
from huggingface_hub import login, hf_hub_download
from huggingface_hub import HfApi
from flask import Flask, request, jsonify
from PIL import Image
from collections import namedtuple
import numpy as np
from matplotlib import colors
import base64
import io
import os

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

# Model
REPO_ID = "mmassonn/CarSegmentation"
MODEL_FILE_NAME = "Segformer_tune.pt"
model_file = hf_hub_download(repo_id=REPO_ID,filename=MODEL_FILE_NAME)
MODEL = torch.load(model_file, map_localisation="cpu", weights_only=False)

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


def generate_img_from_mask(mask, colors_palette=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
    """Genère une image à partir du masque de segmentation."""

    id2category = {0: 'void',
                   1: 'flat',
                   2: 'construction',
                   3: 'object',
                   4: 'nature',
                   5: 'sky',
                   6: 'human',
                   7: 'vehicle'}
    img_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='float')
    for cat in id2category.keys():
        img_seg[:, :, 0] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[0]
        img_seg[:, :, 1] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[1]
        img_seg[:, :, 2] += mask[:, :, cat] * colors.to_rgb(colors_palette[cat])[2]

    return img_seg

def predict_segmentation(
    image_array, 
    image_width: int, 
    image_height: int,
    ):
    '''Genère le masque de couleur à partir du modèle.'''
    image_array = Image.fromarray(image_array).resize((image_width, image_height))
    img_tensor = (torch.from_numpy(img).float().permute(2, 0, 1) / 255.0).unsqueeze(0)
    with torch.no_grad():
        outputs = model(pixel_values=img_tensor.float())
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

@app.route("/predict_mask", methods=['GET', 'POST'])
def segment_image() -> list:
    """Réalise la segmentation d'un images à partir de son path."""
    file = request.get_json(force=True)
    image_file_name = file['file_name']
    image_path = hf_hub_download(
        repo_id=DATASET_IMAGE_REPO_ID,
        filename=image_file_name,
        repo_type="dataset",
        )
    image = Image.open(image_path)
    image_array = np.array(image)

    pred_mask_array = predict_segmentation(
        image_array=image_array,
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
    mask_array = np.array(Image.open(real_mask_path).convert('L'))
    real_mask_array = get_numpy_mask_from_image(mask_img = mask_array)
    real_mask_array_color = generate_img_from_mask(real_mask_array) * 255
    
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
        'real_mask': array_to_base64(real_mask_array_color),
    }
