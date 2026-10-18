"""
Plant Disease Detection - FastAPI Backend (PyTorch)
====================================================
Serves the web UI and provides a /predict endpoint
that classifies uploaded plant leaf images using a PyTorch MobileNetV2 model.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import json
import numpy as np
import uvicorn
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import models, transforms

# ──────────────────────────────────────────────
# Configuration & Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_mobilenetv2.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = 224

# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
# Disease information database
# ──────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_Scab": {
        "description": "Apple scab is caused by the fungus Venturia inaequalis. It leads to dark, scabby lesions on leaves and fruit.",
        "treatment": "Apply fungicides like captan or myclobutanil during spring. Remove fallen infected leaves. Plant resistant varieties.",
        "severity": "moderate"
    },
    "Apple___Black_Rot": {
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. It causes leaf spots, fruit rot, and cankers on branches.",
        "treatment": "Prune dead or infected branches. Apply copper-based fungicides. Remove mummified fruits from the tree.",
        "severity": "high"
    },
    "Apple___Cedar_Apple_Rust": {
        "description": "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae. It produces yellow-orange spots on leaves.",
        "treatment": "Remove nearby cedar/juniper trees if possible. Apply fungicides in spring. Plant resistant apple varieties.",
        "severity": "moderate"
    },
    "Apple___Healthy": {
        "description": "The apple leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, fertilization, and pruning.",
        "severity": "none"
    },
    "Cherry___Healthy": {
        "description": "The cherry leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, fertilization, and pruning.",
        "severity": "none"
    },
    "Cherry___Powdery_Mildew": {
        "description": "Powdery mildew is caused by Podosphaera clandestina. It forms white powdery patches on leaves and shoots.",
        "treatment": "Apply sulfur-based or potassium bicarbonate fungicides. Improve air circulation. Avoid overhead watering.",
        "severity": "moderate"
    },
    "Corn___Common_Rust": {
        "description": "Common rust is caused by Puccinia sorghi. It produces small, reddish-brown pustules on both leaf surfaces.",
        "treatment": "Plant resistant hybrids. Apply foliar fungicides if severe. Early planting can reduce infection.",
        "severity": "moderate"
    },
    "Corn___Gray_Leaf_Spot": {
        "description": "Gray leaf spot is caused by Cercospora zeae-maydis. It creates rectangular gray-brown lesions on leaves.",
        "treatment": "Use resistant hybrids. Practice crop rotation. Apply fungicides at early signs. Tillage to bury debris.",
        "severity": "high"
    },
    "Corn___Healthy": {
        "description": "The corn leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, fertilization, and crop rotation.",
        "severity": "none"
    },
    "Corn___Northern_Leaf_Blight": {
        "description": "Northern leaf blight is caused by Exserohilum turcicum. It creates long, cigar-shaped gray-green lesions.",
        "treatment": "Plant resistant hybrids. Apply strobilurin or triazole fungicides. Practice crop rotation.",
        "severity": "high"
    },
    "Grape___Black_Rot": {
        "description": "Black rot is caused by Guignardia bidwellii. It causes brown circular lesions on leaves and shriveled black berries.",
        "treatment": "Apply mancozeb or myclobutanil fungicides. Remove mummified berries. Ensure good canopy management.",
        "severity": "high"
    },
    "Grape___Esca": {
        "description": "Esca (Black Measles) is a complex disease caused by multiple fungi. It causes tiger-stripe patterns on leaves.",
        "treatment": "No effective chemical control. Remove and destroy infected vines. Protect pruning wounds. Use clean tools.",
        "severity": "high"
    },
    "Grape___Healthy": {
        "description": "The grape leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper canopy management, watering, and pruning.",
        "severity": "none"
    },
    "Grape___Leaf_Blight": {
        "description": "Grape leaf blight (Isariopsis leaf spot) causes dark brown spots with concentric rings on leaves.",
        "treatment": "Apply copper-based fungicides. Remove infected leaves. Improve air circulation through pruning.",
        "severity": "moderate"
    },
    "Peach___Bacterial_Spot": {
        "description": "Bacterial spot is caused by Xanthomonas arboricola. It creates angular, water-soaked lesions on leaves.",
        "treatment": "Apply copper sprays during dormancy. Plant resistant varieties. Avoid overhead irrigation.",
        "severity": "moderate"
    },
    "Peach___Healthy": {
        "description": "The peach leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, fertilization, and pruning.",
        "severity": "none"
    },
    "Pepper___Bacterial_Spot": {
        "description": "Bacterial spot is caused by Xanthomonas campestris. It causes small, dark, water-soaked spots on leaves.",
        "treatment": "Use disease-free seeds. Apply copper-based bactericides. Practice crop rotation. Avoid wet foliage.",
        "severity": "moderate"
    },
    "Pepper___Healthy": {
        "description": "The pepper leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, fertilization, and pest monitoring.",
        "severity": "none"
    },
    "Potato___Early_Blight": {
        "description": "Early blight is caused by Alternaria solani. It creates dark brown spots with concentric rings (target spots).",
        "treatment": "Apply chlorothalonil or mancozeb fungicides. Practice crop rotation. Remove plant debris after harvest.",
        "severity": "moderate"
    },
    "Potato___Healthy": {
        "description": "The potato leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, hilling, and pest monitoring.",
        "severity": "none"
    },
    "Potato___Late_Blight": {
        "description": "Late blight is caused by Phytophthora infestans. It causes large, dark, water-soaked lesions that spread rapidly.",
        "treatment": "Apply preventive fungicides (mancozeb, chlorothalonil). Destroy infected plants immediately. Avoid overhead watering.",
        "severity": "critical"
    },
    "Strawberry___Healthy": {
        "description": "The strawberry leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, mulching, and runner management.",
        "severity": "none"
    },
    "Strawberry___Leaf_Scorch": {
        "description": "Leaf scorch is caused by Diplocarpon earlianum. It produces dark purple spots that merge into scorched areas.",
        "treatment": "Remove infected leaves. Apply captan fungicide. Ensure good air circulation. Avoid overhead watering.",
        "severity": "moderate"
    },
    "Tomato___Bacterial_Spot": {
        "description": "Bacterial spot is caused by Xanthomonas species. It creates small, dark, greasy-looking spots on leaves.",
        "treatment": "Use disease-free transplants. Apply copper sprays. Practice crop rotation. Avoid working with wet plants.",
        "severity": "moderate"
    },
    "Tomato___Early_Blight": {
        "description": "Early blight is caused by Alternaria solani. It produces dark concentric rings (bull's-eye pattern) on lower leaves first.",
        "treatment": "Apply fungicides (chlorothalonil, copper). Mulch to prevent soil splash. Stake plants for airflow. Rotate crops.",
        "severity": "moderate"
    },
    "Tomato___Healthy": {
        "description": "The tomato leaf appears healthy with no signs of disease.",
        "treatment": "Continue regular maintenance: proper watering, staking, pruning, and fertilization.",
        "severity": "none"
    },
    "Tomato___Late_Blight": {
        "description": "Late blight is caused by Phytophthora infestans. It causes large, greasy, dark brown lesions that can destroy entire plants rapidly.",
        "treatment": "Apply preventive fungicides immediately. Remove and destroy all infected plant material. Do not compost infected debris.",
        "severity": "critical"
    },
    # General plant categories (when classified by crop/plant species)
    "Apple": {
        "description": "Identified as Apple (Malus domestica) foliage.",
        "treatment": "Ensure routine monitoring for apple scab, cedar apple rust, and aphids. Maintain proper sunlight and balanced pruning.",
        "severity": "none"
    },
    "Cherry": {
        "description": "Identified as Cherry (Prunus) leaf.",
        "treatment": "Inspect for powdery mildew or leaf spot. Maintain good air circulation and avoid overhead watering.",
        "severity": "none"
    },
    "Corn": {
        "description": "Identified as Corn / Maize (Zea mays) foliage.",
        "treatment": "Check regularly for common rust or blight pustules. Maintain proper soil nutrients and crop rotation.",
        "severity": "none"
    },
    "Grape": {
        "description": "Identified as Grape (Vitis) vine leaf.",
        "treatment": "Ensure canopy airflow and check for downy mildew or black rot. Apply organic fungicides if spots emerge.",
        "severity": "none"
    },
    "Peach": {
        "description": "Identified as Peach (Prunus persica) leaf.",
        "treatment": "Inspect for bacterial spot or peach leaf curl. Apply dormant copper spray in early spring.",
        "severity": "none"
    },
    "Pepper": {
        "description": "Identified as Bell Pepper / Chili (Capsicum) leaf.",
        "treatment": "Monitor for bacterial leaf spot, aphids, and adequate soil calcium. Water at the base.",
        "severity": "none"
    },
    "Potato": {
        "description": "Identified as Potato (Solanum tuberosum) foliage.",
        "treatment": "Keep hills well-covered and monitor closely for early or late blight signs. Practice crop rotation.",
        "severity": "none"
    },
    "Strawberry": {
        "description": "Identified as Strawberry (Fragaria) leaf.",
        "treatment": "Ensure dry foliage and clear old leaves to prevent leaf scorch and gray mold.",
        "severity": "none"
    },
    "Tomato": {
        "description": "Identified as Tomato (Solanum lycopersicum) leaf.",
        "treatment": "Inspect for early/late blight, stake plants for airflow, and water directly at the root zone.",
        "severity": "none"
    }
}


# ──────────────────────────────────────────────
# Model loading helper
# ──────────────────────────────────────────────
def load_pytorch_model(model_path, num_classes, device):
    """Build MobileNetV2 with the same architecture used in training and load weights."""
    model = models.mobilenet_v2(weights=None)

    # Rebuild the same classifier head used during training:
    # The training notebook only replaces classifier[1] with a single Linear layer.
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load saved checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────
# App initialization
# ──────────────────────────────────────────────
app = FastAPI(
    title="Plant Disease Detection API",
    description="Upload a plant leaf image to detect diseases using AI (PyTorch)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static / frontend assets safely if directories exist
if os.path.exists(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
elif os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if os.path.exists(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend_dir")

templates = Jinja2Templates(directory="templates") if os.path.exists(os.path.join(BASE_DIR, "templates")) else None

# Globals
model = None
class_names = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference transform (must match validation transform from training)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model_and_classes():
    """Load the trained PyTorch model and class names safely."""
    global model, class_names
    if model is not None and class_names is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
        print("  Run the training notebook first to train and save the model.")
        return False

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

        # Extract class names from checkpoint (training notebook embeds them)
        if isinstance(checkpoint, dict) and 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            print(f"[OK] Loaded {len(class_names)} class names from checkpoint")
        elif os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"[OK] Loaded {len(class_names)} class names from {CLASS_NAMES_PATH}")
        else:
            print(f"[WARNING] Class names not found in checkpoint or {CLASS_NAMES_PATH}")
            return False

        num_classes = checkpoint.get('num_classes', len(class_names))
        model = load_pytorch_model(MODEL_PATH, num_classes, device)
        print(f"[OK] PyTorch model loaded from: {MODEL_PATH}")
        print(f"  Device: {device}")
        print(f"  Classes: {class_names}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False


@app.on_event("startup")
async def startup_load_model():
    """Load the trained PyTorch model and class names on startup."""
    load_model_and_classes()


@app.get("/")
async def home(request: Request):
    """Serve the main page from frontend directory."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    if templates and os.path.exists(os.path.join(BASE_DIR, "templates", "index.html")):
        return templates.TemplateResponse("index.html", {"request": request})
    return JSONResponse(status_code=404, content={"detail": "Frontend index.html not found"})


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_loaded": class_names is not None,
        "num_classes": len(class_names) if class_names else 0,
        "device": str(device)
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict disease from an uploaded plant leaf image.
    Returns top 5 predictions with confidence scores and disease info.
    """
    global model, class_names

    # Auto-load model on-demand if needed
    if model is None or class_names is None:
        load_model_and_classes()

    # Validate model is loaded
    if model is None or class_names is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please verify the model file in the model/ directory."
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not an image. Please upload a JPG, PNG, or BMP image."
        )

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = inference_transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        predictions = probabilities.cpu().numpy()

        # Get top 5 predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        results = []
        for idx in top_indices:
            class_name = class_names[idx]
            parts = class_name.split("___")
            plant = parts[0]
            disease = parts[1].replace("_", " ") if len(parts) > 1 else "Plant Foliage"

            # Look up disease info
            info_key = class_name.replace(" ", "_")
            disease_data = DISEASE_INFO.get(info_key, {
                "description": "No information available for this condition.",
                "treatment": "Consult a local agricultural extension service for guidance.",
                "severity": "unknown"
            })

            results.append({
                "class_name": class_name,
                "plant": plant,
                "disease": disease,
                "confidence": float(predictions[idx]) * 100,
                "description": disease_data["description"],
                "treatment": disease_data["treatment"],
                "severity": disease_data["severity"]
            })

        return JSONResponse(content={
            "success": True,
            "predictions": results,
            "top_prediction": results[0] if results else None
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


# Mount root frontend directory for static assets (style.css, app.js, etc.)
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_root")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
