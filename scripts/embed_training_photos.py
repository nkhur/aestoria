from pymongo import MongoClient
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
from io import BytesIO

from dotenv import load_dotenv
import os
load_dotenv()

client = MongoClient(os.getenv('MONGO_URL'))
db = client["aestoria_app"]
collection = db["training_images"]

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for doc in collection.find():
    url = doc.get("url")
    if not url:
        continue

    try:
        img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True) # normalized euclidean distance, so that we can train on angle for cosine-similiarity
            emb_list = emb.squeeze().cpu().tolist()

        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"url": url, "clip_embedding": emb_list}},
            upsert=True
        )

        print(f"Stored embedding for {url}:")

    except Exception as e:
        print(f"Error {url}: {e}")
