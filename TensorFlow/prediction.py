import cv2
import numpy as np
import pickle


model = pickle.load(open("./model.pkl", "rb"), encoding="latin1")


def predict(path="../archive/brain_tumor_dataset/no/15 no.jpg"):
    image_path = path
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    prediction = model.predict(img)
    prediction = prediction.squeeze()
    print(prediction)

    if prediction[0] > prediction[1]:
        return "There is no Brain Tumor"
    else:
        return "There is a Brain Tumor"


print(predict())

# /////////////////////////Fast Api/////////////////////////////////

from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://127.0.0.1:3000/input",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/get")
async def intro():
    return {"message": "Welcome to the Prediction of Brain Tumor"}


@app.post("/input")
async def create_item(input_data: dict):
    try:
        s = predict()
        return {"message": s}
    except KeyError:
        raise HTTPException(
            status_code=422, detail="Missing 'input' field in the request body"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
