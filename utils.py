import io

import numpy as np
import torch
from PIL import Image
from fastapi import UploadFile
from torchvision import transforms

from model import vision_model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
classes = ['Adobong Manok', 'Inasal na Manok', 'Sinigang', 'Sisig', 'Tortang Talong']

async def img2tensor(image: UploadFile):
    byte_image = io.BytesIO(await image.read())
    image = Image.open(byte_image)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return transform(image).to(device)

async def classify_image(image: UploadFile):
    tensor_image = await img2tensor(image)
    tensor_image = tensor_image.unsqueeze(0)
    vision_model.eval()
    with torch.no_grad():
        result = vision_model(tensor_image)
    return classes[result.argmax(dim=1)]