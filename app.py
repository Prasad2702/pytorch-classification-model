import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the same model structure used during training
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),         # Binary classification
            nn.Sigmoid(),              # Keep Sigmoid for BCELoss model
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load("pen_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# Define transform (must match validation transform during training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🖊️ Pen vs Not-Pen Classifier")
st.write("Upload an image and the model will predict whether it's a pen or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform and predict
    img = transform(image).unsqueeze(0)  # Add batch dimension

    with st.spinner("Predicting..."):
    # run model

    with torch.no_grad():
        output = model(img)
        prediction = (output > 0.5).float().item()

    label = "Pen 🖊️" if prediction == 1.0 else "Not a Pen ❌"
    st.subheader(f"Prediction: {label}")
