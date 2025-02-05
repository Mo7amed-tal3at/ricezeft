import pickle
import re
from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from tensorflow.keras.models import load_model


__version__ = "0.1.0"

# Define base directory and load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct path for the model file
model_path = os.path.join(BASE_DIR, "rice.h5")

# Load the model
model = load_model(model_path)



classes = ['Rice___Brown_Spot','Rice___Healthy','Rice___Leaf_Blast','Rice___Neck_Blast']

# Helper function to load and preprocess a single image
def process_and_predict(img):
    img = img.resize((256, 256))
    img_array = img_to_array(img)  # 
    img_array = np.expand_dims(img_array, axis=0)  # 
    predictions = model.predict(img_array)
    res = classes[np.argmax(predictions)]
    return res