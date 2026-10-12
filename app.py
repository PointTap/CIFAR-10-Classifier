import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import altair as alt
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
top_classes = [classes[idx] for idx in top_idxs]

# Create DataFrame and sort by probability descending
df = pd.DataFrame({
    "Class": top_classes,
    "Probability": top_probs.tolist()
}).sort_values("Probability", ascending=False)

# Altair bar chart
bars = alt.Chart(df).mark_bar().encode(
    x=alt.X("Class:N", sort="-y", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Probability:Q"),
    tooltip=["Class", "Probability"]
).properties(width=500, height=300)

# Add probability numbers BELOW each class name
labels_below = alt.Chart(df).mark_text(
    align="center",
    baseline="top",
    dy=30,  # push text below x-axis labels
    color="black"
).encode(
    x="Class:N",
    y=alt.value(0),  # baseline
    text=alt.Text("Probability:Q", format=".2f")
)

st.write("### Top-5 class probabilities (sorted bar chart):")
st.altair_chart(bars + labels_below, use_container_width=True)
