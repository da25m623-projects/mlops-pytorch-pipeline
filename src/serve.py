import os
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from model import get_model


MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        "/app/checkpoints/classifier_v1.pt",
    )
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_model():
    """Load the trained model checkpoint."""

    model = get_model(
        architecture="resnet18",
        num_classes=10,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = None
model_load_error = None

try:
    model = load_model()
except Exception as exc:
    model_load_error = str(exc)


app = FastAPI(
    title="CIFAR-10 Model Serving API",
    version="1.0.0",
)


preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


@app.get("/health")
def health():
    """Return service health status."""

    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": model_load_error,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """Return CIFAR-10 class probabilities for an uploaded image."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        image_bytes = await image.read()

        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        ) from exc

    tensor = preprocess(pil_image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    result = {
        CLASS_NAMES[index]: round(
            float(probabilities[index]),
            6,
        )
        for index in range(len(CLASS_NAMES))
    }

    predicted_index = int(
        probabilities.argmax().item()
    )

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "probabilities": result,
    }
