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
    model = BayesianCNN()
    state_dict = torch.load("bayesian_cnn.pth", map_location="cpu")
    model.load_state_dict(state_dict)
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

# Show available classes at upload section
st.write("### The model can classify into these categories:")
st.write(", ".join(classes))

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Default image selection UI
st.write("Or choose one of the default images:")

col1, col2, col3 = st.columns(3)
default_choice = None

with col1:
    img1 = Image.open("static/images1.jpg")
    st.image(img1.resize((200, 200)), caption="images1.jpg")
    if st.button("Use images1", key="btn1"):
        default_choice = "images1.jpg"

with col2:
    img2 = Image.open("static/images2.jpg")
    st.image(img2.resize((200, 200)), caption="images2.jpg")
    if st.button("Use images2", key="btn2"):
        default_choice = "images2.jpg"

with col3:
    img3 = Image.open("static/images3.jpg")
    st.image(img3.resize((200, 200)), caption="images3.jpg")
    if st.button("Use images3", key="btn3"):
        default_choice = "images3.jpg"

# Decide which image to use
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif default_choice is not None:
    image = Image.open(f"static/{default_choice}").convert("RGB")
else:
    st.info("Upload an image or click one of the defaults above.")
    st.stop()

# Show chosen image
st.image(image, caption="Chosen Image", use_column_width=True)

# Preprocess
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    predicted_class = probs.argmax(dim=1).item()

st.write(f"Predicted class: **{classes[predicted_class]}**")

# Show Top-5 probabilities sorted
top_probs, top_idxs = torch.topk(probs[0], 5)
st.write("Top-5 class probabilities:")
for i in range(top_probs.size(0)):
    st.write(f"{classes[top_idxs[i]]}: {top_probs[i].item():.4f}")
