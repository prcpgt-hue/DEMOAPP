
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import pickle
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = CNN()
model.to(device)

model_path_pkl = './saved_models/mnist_cnn_model.pkl'

try:
    with open(model_path_pkl, 'rb') as f:
        loaded_model = pickle.load(f)
    loaded_model.eval() # Set the model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the 'CNN' class definition is available and the model file exists.")
    print("Attempting to load state_dict if full model load fails")
    # Fallback to loading state_dict if pickle.load fails on full model
    try:
        model_weights_path_pth = './saved_models/mnist_cnn_weights.pth'
        model.load_state_dict(torch.load(model_weights_path_pth, map_location=device))
        loaded_model = model
        loaded_model.eval()
        print("Model weights loaded successfully!")
    except Exception as sd_e:
        print(f"Error loading model weights: {sd_e}")
        loaded_model = None # Indicate failure to load any model




app = FastAPI()

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class PredictionResult(BaseModel):
    predicted_digit: int
    confidence: float

@app.post("/predict", response_model=PredictionResult)
async def predict_image(file: UploadFile = File(...)):
    if loaded_model is None:
        return {"error": "Model not loaded"}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = loaded_model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_digit = probabilities.argmax().item()
    confidence = probabilities[predicted_digit].item()

    return {"predicted_digit": predicted_digit, "confidence": confidence}

@app.get("/")
async def root():
    return {"message": "MNIST CNN Prediction API. Go to /docs for API documentation."}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

