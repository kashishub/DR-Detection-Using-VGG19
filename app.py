import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("dr_model.h5")

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        result = f"Diabetic Retinopathy Detected (Confidence: {prediction:.2f})"
    else:
        result = f"No Diabetic Retinopathy (Confidence: {1 - prediction:.2f})"

    return result

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Diabetic Retinopathy Detection System",
    description="Upload a retinal fundus image to detect Diabetic Retinopathy."
)

if __name__ == "__main__":
    interface.launch()