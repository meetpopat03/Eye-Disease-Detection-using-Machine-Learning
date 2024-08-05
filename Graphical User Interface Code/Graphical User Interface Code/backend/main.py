

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = r"d:\Virtual_Desktop\Desktop\Tech_Work\potato_disease_ml_project\Eye_disease_Detection\Model2\api\conv2d_6-Eye_Disease-95.30.h5"

if os.path.exists(model_path):
    MODEL = tf.keras.models.load_model(model_path)
else:
    print(f"Error: Model file '{model_path}' not found.")

# MODEL = tf.keras.models.load_model(r"d:\Virtual_Desktop\Desktop\Tech_Work\potato_disease_ml_project\Eye_disease_Detection\Model2\efficientnetb3-Eye-Disease-93.46.h5")

CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glucoma", "Normal"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def resize_image_pil(input_image_array, new_size=(224, 224)):
    resized_image = cv2.resize(input_image_array, new_size)
    # You can now work with 'resized_image' without saving it to disk
    return resized_image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image = read_file_as_image(await file.read())
    image = resize_image_pil(image)
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)