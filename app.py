import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Title
st.title("CIFAR-10 Image Classifier")

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    # Adjust path to your trained model file
    model = torch.load("bayesian_cnn.pth", map_location="cpu")
    model.eval()
    return model

model = load_model()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # CIFAR-10 class labels
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    st.write(f"Predicted class: **{classes[predicted_class]}**")
