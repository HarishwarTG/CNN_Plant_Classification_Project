import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, img, labels):
    img = preprocess(img)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    return labels[predicted_class]

def load_model_and_labels():
    model_path = r'Z:\CNN_Crop_Classification\Notebook\plant_classification_tf'
    model = load_model(model_path)
    
    labels = {
        0: 'Unlabeled',
        1: 'basil',
        2: 'blueash',
        3: 'boxelder',
        4: 'cilantro',
        5: 'daisy',
        6: 'mint',
        7: 'oak leaves',
        8: 'oregano',
        9: 'parsley',
        10: 'poison ivy',
        11: 'poison oak',
        12: 'rose',
        13: 'tulip'
    }
    
    return model, labels
