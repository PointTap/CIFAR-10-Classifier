import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from model import BayesianCNN   # make sure your model class is defined in model.py

# Title
st.title("CIFAR-10 Image Classifier")

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    model = BayesianCNN()  # instantiate your model architecture
    state_dict = torch.load("bayesian_cnn.pth", map_location="cpu")
    model.load_state_dict(state_dict)  # load weights
    model.eval()
    return model

model = load_model()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 class labels
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Default sample if nothing uploaded
if uploaded_file is None:
    st.info("Upload an image to classify. Using default sample...")
    image = Image.open("static/sample.jpg").convert("RGB")  # put a sample image in static/
else:
    image = Image.open(uploaded_file).convert("RGB")

# Show image
st.image(image, caption="Input Image", use_column_width=True)

# Preprocess
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)  # convert logits to probabilities
    predicted_class = probs.argmax(dim=1).item()

st.write(f"Predicted class: **{classes[predicted_class]}**")

# Show Top-5 probabilities sorted
top_probs, top_idxs = torch.topk(probs[0], 5)
st.write("Top-5 class probabilities:")
for i in range(top_probs.size(0)):
    st.write(f"{classes[top_idxs[i]]}: {top_probs[i].item():.4f}")
