import os
import gdown
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# ------------------------------
# Download model from Google Drive (only if not exists)
# ------------------------------

MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1-1ZyhBrGSCaTSTEW561QWRWwHdo8LqAy"
    gdown.download(url, MODEL_PATH, quiet=False)

# ------------------------------
# Load Model
# ------------------------------

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = (224, 224)

def predict(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        label = "Diabetic Retinopathy Detected"
        confidence = float(prediction)
    else:
        label = "No Diabetic Retinopathy"
        confidence = float(1 - prediction)

    return f"{label} (Confidence: {confidence:.4f})"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Diabetic Retinopathy Detection (VGG19)",
    description="Upload a retinal fundus image to detect DR."
)

# ------------------------------
# Render Port Binding
# ------------------------------

port = int(os.environ.get("PORT", 7860))

interface.launch(server_name="0.0.0.0", server_port=port)