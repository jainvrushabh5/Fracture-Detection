import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# Load the model
model_transfer = models.resnet50(pretrained=False)
model_transfer.fc = torch.nn.Linear(model_transfer.fc.in_features, 2)
model_transfer.load_state_dict(torch.load('fracture_detection_model.pth', map_location=torch.device('cpu')))
model_transfer.eval()

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Streamlit app
st.title("Fracture Detection")
st.write("Upload an X-ray image to check if it contains a fracture.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model_transfer(image_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["Non-fractured", "Fractured"]
    prediction = class_names[predicted.item()]
    st.write(f"Prediction: {prediction}")
