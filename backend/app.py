import tensorflow as tf
from tensorflow.keras.models import load_model
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
MODEL_FILE_NAME = "mobilenet_unet_categorical_crossentropy_augFalse.keras"
model_file = hf_hub_download(repo_id=REPO_ID,filename=MODEL_FILE_NAME)
MODEL = load_model(model_file, compile=False)

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

MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 128


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

def get_numpy_mask_from_image(mask_img):
    mask_array   = np.zeros((mask_img.shape[0], mask_img.shape[1], 8),dtype=int) # create a mask with zeros
    Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )

    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]

    #--------------------------------------------------------------------------------
    # Create dictionaries for a fast lookup
    #--------------------------------------------------------------------------------

    # Please refer to the main method below for example usages!

    # name to label object
    name2label      = { label.name    : label for label in labels           }
    # id to label object
    id2label        = { label.id      : label for label in labels           }
    # trainId to label object
    id2category     = { label[4]   : label.category for label in labels  }
    trainId2label   = { label.trainId : label for label in reversed(labels) }
    # category to list of label objects
    category2labels = {}
    for label in labels:
        category = label.category
        if category in category2labels:
            category2labels[category].append(label)
        else:
            category2labels[category] = [label]
    for k, v in category2labels.items():
        #print("[INFO] : Processing for main category ", k)
        for category_label in v: 
            categoryId = category_label[4]
            labelID = category_label[1]
            #print("    [INFO] : Processing for subcategory {} (labelID = {} | categoryId={})".format(category_label[0], labelID, categoryId))
            mask_array[:,:,categoryId] = np.logical_or(mask_array[:,:,categoryId],(mask_img==labelID))
    
    return mask_array

def predict_segmentation(image_array, image_width, image_height):
    '''Genère le masque de couleur à partir du modèle.'''

    image_array = Image.fromarray(image_array).resize((image_width, image_height))
    image_array = np.expand_dims(np.array(image_array), axis=0)
    mask_predict = MODEL.predict(image_array)
    mask_predict = np.squeeze(mask_predict, axis=0)
    mask_color = generate_img_from_mask(mask_predict) * 255

    return mask_color


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
    image_path = hf_hub_download(repo_id=DATASET_IMAGE_REPO_ID, filename=image_file_name, repo_type="dataset")
    image = Image.open(image_path)
    image_array = np.array(image)

    pred_mask_array = predict_segmentation(image_array=image_array, image_width=MODEL_INPUT_WIDTH,
                                           image_height=MODEL_INPUT_HEIGHT)
    _ , dataset_file_paths = get_dataset_file_path()
    real_mask_file_name = dataset_file_paths[image_file_name]
    real_mask_path = hf_hub_download(repo_id=DATASET_MASK_REPO_ID, filename=real_mask_file_name, repo_type="dataset")
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
