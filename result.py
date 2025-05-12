import os
import torch
import pandas as pd
import torchvision
import streamlit as st
from PIL import Image
from dataset import vit_101_transform
from ViT import ViT
from TransferLearning import create_ViT_model

# Constants
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load nutrition data
nutrition_df = pd.read_csv('nutrition.csv')

# Load class labels
with open('classes.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

NON_FOOD_ITEMS = [
    'person', 'face', 'landscape', 'building', 'car', 'vehicle', 'animal', 'dog', 'cat', 'bird',
    'flower', 'tree', 'plant', 'furniture', 'electronics', 'phone', 'computer', 'clothing',
    'shoes', 'device', 'toy', 'book', 'text', 'document', 'road', 'sky', 'mountain', 'water',
    'river', 'ocean', 'beach', 'snow'
]

def is_food_item(class_name, confidence):
    if any(non_food in class_name.lower() for non_food in NON_FOOD_ITEMS):
        return False
    if confidence < 3:
        return False
    if not any(nutrition_df['label'] == class_name):
        return False
    return True

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vit_model, _ = create_ViT_model(classes=len(class_labels), seed=42)
    state_dict = torch.load('models/food101_vit_model.pth', map_location=device)
    vit_model.load_state_dict(state_dict)
    vit_model.eval()
    vit_model = vit_model.to(device)
    return vit_model, device

model, device = load_model()

def classify_image(image: Image.Image):
    image_tensor = vit_101_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_class_idx].item() * 100
        return pred_class_idx, confidence

# Streamlit UI
st.title("🍱 Food Classification and Nutrition Info with ViT")

# Dropdown to choose image input type
input_type = st.selectbox("Choose input type", ["Upload Image", "Use Camera"])

image = None
filename = None

if input_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload a food image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        filename = uploaded_file.name

elif input_type == "Use Camera":
    camera_image = st.camera_input("Take a picture")
    if camera_image:
        image = Image.open(camera_image).convert('RGB')
        filename = "camera_capture.jpg"

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    pred_class_idx, confidence = classify_image(image)
    pred_class = class_labels[pred_class_idx]

    st.markdown(f"**Predicted Class**: `{pred_class.replace('_', ' ').title()}`")
    st.markdown(f"**Confidence**: `{confidence:.2f}%`")

    if not is_food_item(pred_class, confidence):
        st.error("🚫 This image does not appear to contain food.")
    else:
        st.success("✅ This image likely contains food!")

        food_rows = nutrition_df[nutrition_df['label'] == pred_class]
        if not food_rows.empty:
            st.subheader("🍽️ Nutritional Information")
            for i, row in food_rows.iterrows():
                st.markdown(f"**Serving Size**: {row['weight']}g")
                st.markdown(f"- Calories: {row['calories']}kcal")
                st.markdown(f"- Protein: {row['protein']}g")
                st.markdown(f"- carbohydrates: {row['carbohydrates']}g")
                st.markdown(f"- Fats: {row['fats']}g")
                st.markdown(f"- Fiber: {row['fiber']}g")
                st.markdown(f"- Sugars: {row['sugars']}g")
                st.markdown(f"- Sodium: {row['sodium']}mg")

