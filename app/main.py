from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# إنشاء التطبيق
app = FastAPI()

# السماح بالطلبات من واجهة الويب
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

# تحميل النموذج
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "rice.h5")

if os.path.exists(model_path):
    MODEL = tf.keras.models.load_model(model_path)
    CLASS_NAMES = ['Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast']
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# مسار الصفحة الرئيسية
@app.get("/")
async def root():
    return {"message": "Welcome to the Rice Disease Prediction API!"}

# اختبار إذا كان التطبيق يعمل
@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    image = np.array(image) 
    image = image/255 
    return image
# مسار التنبؤ
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions,axis=1)[0]]
    confidence = float(np.max(predictions,axis=1)[0])

    return {
        'class': predicted_class,
        'confidence': confidence
    }

# تشغيل التطبيق
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
