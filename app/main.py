# Main FastAPI application file
import io
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from PIL import Image

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load pre-trained model
model_path = './models/image_classification_model_vgg16.h5'
model = tf.keras.models.load_model(model_path)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['0', '1', '2', '3', '4']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) # Create a batch
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)

    # VGG16 Prediction
    predictions_vgg16 = model.predict(processed_image)
    score_vgg16 = tf.nn.softmax(predictions_vgg16[0])

    # Predict class
    predicted_class = CLASS_NAMES[np.argmax(score_vgg16)]
    confidence = np.max(score_vgg16) * 100

    # Prepare results for template
    results_vgg16 = {CLASS_NAMES[i]: f"{score_vgg16[i] * 100:.2f}%" for i in range(len(CLASS_NAMES))}

    return {
        "vgg16": results_vgg16,
        "final_prediction": {
            "class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
    }
