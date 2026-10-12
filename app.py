import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import BayesianCNN   # import your model class

st.title("CIFAR-10 Image Classifier")

@st.cache_resource
def load_model():
    model = BayesianCNN()  # create the model architecture
    state_dict = torch.load("bayesian_cnn.pth", map_location="cpu")
    model.load_state_dict(state_dict)  # load weights
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    st.write(f"Predicted class: **{classes[predicted_class]}**")
