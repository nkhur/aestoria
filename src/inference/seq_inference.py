# src/inference/run_inference_urls.py
import torch
from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms
from src.models.sequence_autoencoder import SeqAutoencoder
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# --- Config ---
MODEL_PATH = "src/models/autoencoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_SEQ = 3  # must match training

# --- Load CLIP model ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Load trained autoencoder ---
autoencoder = SeqAutoencoder(emb_dim=512, k=K_SEQ).to(DEVICE)
autoencoder.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
autoencoder.eval()

# --- Preprocess ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Load images from URLs ---
def load_images_from_urls(urls):
    images, filenames = [], []
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
            filenames.append(f"image_{i+1}")
        except Exception as e:
            print(f"Failed to load {url}: {e}")
    return images, filenames

# --- CLIP embeddings ---
def get_clip_embeddings(images):
    inputs = clip_processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb

# --- Sequence images using AE predictions ---
def sequence_images(embeddings, filenames, k=K_SEQ):
    embeddings_np = embeddings.cpu().numpy()
    N = len(embeddings_np)
    if N <= k:
        return filenames  # not enough images to predict sequence

    # Start with first k images as seed
    ordered_idx = list(range(k))
    remaining = set(range(k, N))

    while remaining:
        # Prepare input: last k embeddings
        last_k_emb = embeddings_np[ordered_idx[-k:]].reshape(1, k, -1)
        last_k_emb_t = torch.tensor(last_k_emb, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_next = autoencoder(last_k_emb_t)  # predicted next embedding

        # Compare predicted embedding with remaining
        sims = []
        for idx in remaining:
            sim = (pred_next.cpu().numpy() @ embeddings_np[idx].reshape(-1, 1)).item()
            sims.append(sim)

        next_idx = list(remaining)[np.argmax(sims)]
        ordered_idx.append(next_idx)
        remaining.remove(next_idx)

    return [filenames[i] for i in ordered_idx]

# --- Main ---
if __name__ == "__main__":
    urls =  [
        "https://i.pinimg.com/1200x/d8/13/1a/d8131a26f031f70ff5c5710e628b5a21.jpg",
        "https://i.pinimg.com/1200x/82/3d/24/823d245b5d32a1b1f1223a17aa6bd5c3.jpg",
        "https://i.pinimg.com/736x/56/1b/7f/561b7f29ceb702c132363f12411ab238.jpg",
        "https://i.pinimg.com/736x/8e/06/64/8e0664ade0c14bb00a6ef10c8b47f0ee.jpg",
        "https://i.pinimg.com/736x/d6/2c/63/d62c63671d56c8a849404b5c642c722d.jpg",
        "https://i.pinimg.com/736x/05/2f/eb/052feb8f1aae5e49fe0350c8e60bc4f3.jpg",
        "https://i.pinimg.com/736x/0f/1b/9e/0f1b9e5f6bd520bef91c7818d45a76c0.jpg",
        "https://i.pinimg.com/736x/7e/39/0d/7e390d1c1a057ea481c4a20d025f8207.jpg",
        "https://i.pinimg.com/736x/dc/60/58/dc605861947147c0257093296c2c34bb.jpg",
        "https://i.pinimg.com/1200x/ba/a9/3d/baa93d5bd23d71a2d1786b224e979604.jpg"
    ]
    images, filenames = load_images_from_urls(urls)
    if not images:
        print("No images loaded from URLs!")
        exit()

    # Step 1: CLIP embeddings
    clip_embeddings = get_clip_embeddings(images)

    # Step 2: Sequence images using autoencoder prediction
    ordered_files = sequence_images(clip_embeddings, filenames, k=K_SEQ)

    print("\nSuggested dump order:")
    for i, fname in enumerate(ordered_files, 1):
        print(f"{i}. {fname}")


# [
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/505393478_18510907228041919_846626324336531502_n.jpg?stp=dst-jpg_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5zZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=1DEiHtqL6gwQ7kNvwHcn_Hn&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNDk4OTIzMDU3NA%3D%3D.3-ccb7-5&oh=00_AfW42Pfwbl0iE4liLpC6ixhwublSJeQuIGXTCtIk3zHkkQ&oe=68AAA174&_nc_sid=22de04",
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/505471482_18510907264041919_5367835217550650924_n.jpg?stp=dst-jpegr_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5oZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=FSdSB7dxlg0Q7kNvwETBD6W&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNTAyMzAwMTA4NA%3D%3D.3-ccb7-5&oh=00_AfV27E62pIRSb1tslzUxQgRhELwpWJf1gEJdPUk8dvimRw&oe=68AA909C&_nc_sid=22de04",
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/504842802_18510907252041919_6915379935689138060_n.jpg?stp=dst-jpegr_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5oZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=oLhMJUI5utkQ7kNvwGJWvuG&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNDk4OTQyNTE0Ng%3D%3D.3-ccb7-5&oh=00_AfUGK-1mtDmXuAIiZOlcA_FIdYSUwWcQOeGh82LWfA0k_Q&oe=68AA8E2A&_nc_sid=22de04",
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/504395045_18510907273041919_2963493069842395308_n.jpg?stp=dst-jpegr_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5oZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=1RUhW2oOWQMQ7kNvwFJMyGZ&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNDk3MjYxNzE0NQ%3D%3D.3-ccb7-5&oh=00_AfXuu9Yt-FJNGd8b_UPhG0Gx3OzQPKpSsh9CZdEXNJuZ4Q&oe=68AA8340&_nc_sid=22de04",
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/504146056_18510907183041919_1548178644514812002_n.jpg?stp=dst-jpegr_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5oZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=mFfTNv8sxjMQ7kNvwGQgzIT&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNDk5NzcwMDM4MQ%3D%3D.3-ccb7-5&oh=00_AfVhAOP1BI6jqlfrCx2R9--bNN6Mb_ZDY7fI7lz5gXZ2Cw&oe=68AA6DFE&_nc_sid=22de04",
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/504151386_18510907237041919_5508467561361404752_n.jpg?stp=dst-jpegr_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5oZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=xa34zRo2angQ7kNvwEfALJa&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNDkyMjEzODI2Mw%3D%3D.3-ccb7-5&oh=00_AfUtSlKywOpFEFDfQmfch1Ric3CWeoLiCaY7eUfqXpnDFA&oe=68AA6C14&_nc_sid=22de04",
#         "https://instagram.fdel27-5.fna.fbcdn.net/v/t51.2885-15/505437545_18510907255041919_1621568670051526500_n.jpg?stp=dst-jpegr_e35_tt6&efg=eyJ2ZW5jb2RlX3RhZyI6IkNBUk9VU0VMX0lURU0uaW1hZ2VfdXJsZ2VuLjE0NDB4MTQzOS5oZHIuZjc1NzYxLmRlZmF1bHRfaW1hZ2UuYzIifQ&_nc_ht=instagram.fdel27-5.fna.fbcdn.net&_nc_cat=107&_nc_oc=Q6cZ2QFqj2B4dTSP7YAdn76fTEB8bkgCEEpjlu2ChVtc3L3X_k4jnn_-P9E95j_W1cJumE0&_nc_ohc=hAjJvoTkYiAQ7kNvwF7qr4T&_nc_gid=HOhIIMLUxHNLATfZy2Pm7w&edm=APoiHPcBAAAA&ccb=7-5&ig_cache_key=MzY1MDU2NDEyNDk3MjQ2NDAxMA%3D%3D.3-ccb7-5&oh=00_AfX_QM93H-lUMrwJn6fYiC1DJqxZcaFGf0nNozYr8wFmTQ&oe=68AA6E3A&_nc_sid=22de04"
#     ]

'''
[
        "https://i.pinimg.com/236x/64/1c/71/641c71bdf94f5de197602d719ffbcea8.jpg",
        "https://i.pinimg.com/236x/0c/20/87/0c2087dcb8d041c0b802cf0af92cc36b.jpg",
        "https://i.pinimg.com/1200x/f0/ca/9e/f0ca9e532a14443f1bfd50550b6d41af.jpg",
        "https://i.pinimg.com/1200x/f0/94/db/f094dbdb7b3a13b0306b86f316ba1526.jpg",
        "https://i.pinimg.com/736x/97/33/bc/9733bce0efc10f5193ac5a22ca7aabd9.jpg",
        "https://i.pinimg.com/736x/2e/b3/e2/2eb3e217613561dccb9f415556012e98.jpg",
        "https://i.pinimg.com/236x/a6/8b/e3/a68be398699ea410f1d6f679266c1ab1.jpg",
    ]
'''