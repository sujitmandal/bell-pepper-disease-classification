# Author : Sujit Mandal
import uvicorn
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Github: https://github.com/sujitmandal
# Pypi : https://pypi.org/user/sujitmandal/
# LinkedIn : https://www.linkedin.com/in/sujit-mandal-91215013a/

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


endpoint = 'http://localhost:8501/v1/models/bell_pepper_model:predict'

class_name = ['Bacterial spot', 'healthy']

def read_image(data) -> np.array:
    image = np.array(Image.open(BytesIO(data)))

    return(image)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": image_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)

    predictions = np.array(response.json()["predictions"][0])

    predicted_class = class_name[np.argmax(predictions)]

    accuracy = np.max(predictions)

    result = {
        'class' : predicted_class,
        'accuracy' : float(accuracy)
    }

    return(result)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)
