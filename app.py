import os
import torch
import torch.nn.functional as F
from flask import Flask, request, render_template
from torchvision import transforms
from PIL import Image
from io import BytesIO
from model import BayesianCNN

# CIFAR-10 classes
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# Load model weights
net = BayesianCNN()
net.load_state_dict(torch.load("bayesian_cnn.pth", map_location="cpu"))
net.eval()

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

@app.route("/", methods=["GET", "POST"])
def index():
    preview_file = None
    error = None
    results = None

    if request.method == "POST":
        # Case 1: User clicked one of the default images
        if "default_image" in request.form:
            filename = request.form["default_image"]
            default_path = os.path.join("static", filename)
            try:
                img = Image.open(default_path).convert("RGB")
                preview_file = filename
            except Exception:
                error = f"Default image {filename} not valid"
                return render_template("index.html", results=None, error=error, classes=classes)
            img = transform(img).unsqueeze(0)

        # Case 2: User uploaded an image
        else:
            file = request.files.get("image")
            if not file or file.filename == "":
                error = "No image uploaded"
                return render_template("index.html", results=None, error=error, classes=classes)

            try:
                img_bytes = file.read()
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                preview_file = file.filename
            except Exception:
                error = "Invalid or unsupported image file"
                return render_template("index.html", results=None, error=error, classes=classes)

            img = transform(img).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            outputs = net(img)
            probs = F.softmax(outputs, dim=1)[0]

        results = {classes[i]: float(probs[i]) for i in range(10)}

    return render_template("index.html", results=results, preview=preview_file, error=error, classes=classes)

if __name__ == "__main__":
    app.run(debug=True)
