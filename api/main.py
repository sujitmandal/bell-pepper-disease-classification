# Author : Sujit Mandal
import uvicorn
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



# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )



model = tf.keras.models.load_model("../saved_models/2")

class_name = ['Bacterial spot', 'healthy']


def read_image(data) -> np.array:
    image = np.array(Image.open(BytesIO(data)))

    return(image)


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    predictions = model.predict(image_batch)

    predicted_class = class_name[np.argmax(predictions[0])]

    accuracy = np.max(predictions[0])

    result = {
        'class' : predicted_class,
        'accuracy' : float(accuracy)
    }

    return(result)


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=5000)
