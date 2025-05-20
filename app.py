import os
import torch
import pandas as pd
import torchvision
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dataset import vit_101_transform
from ViT import ViT
from TransferLearning import create_ViT_model
import io
import requests

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load nutrition data
nutrition_df = pd.read_csv('nutrition.csv')

# Load class labels
with open('classes.txt', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# List of non-food items that might be incorrectly classified
NON_FOOD_ITEMS = [
    'person', 'face', 'landscape', 'building', 'car', 'vehicle', 'animal', 'dog', 'cat', 'bird',
    'flower', 'tree', 'plant', 'furniture', 'electronics', 'phone', 'computer', 'clothing',
    'shoes', 'device', 'toy', 'book', 'text', 'document', 'road', 'sky', 'mountain', 'water',
    'river', 'ocean', 'beach', 'snow'
]

# Function to check if the image contains food
def is_food_item(class_name, confidence):
    # Check if the predicted class is in our non-food list
    for non_food in NON_FOOD_ITEMS:
        if non_food in class_name.lower():
            return False
    
    # If confidence is very low, it might not be food
    if confidence < 3:
        return False
    
    # Check if we have nutrition data for this class
    if not any(nutrition_df['label'] == class_name):
        return False
        
    return True

# Function to load the model directly from Hugging Face without saving locally
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model architecture
    vit_model, _ = create_ViT_model(classes=len(class_labels), seed=42)

    # Model URL (raw file)
    remote_model_url = 'https://huggingface.co/spaces/officialamit558/vitmodel/resolve/main/food101_vit_model.pth'

    print("Downloading model from Hugging Face...")
    response = requests.get(remote_model_url)
    response.raise_for_status()  # Ensure it downloaded correctly

    # Load model weights from in-memory buffer
    buffer = io.BytesIO(response.content)
    state_dict = torch.load(buffer, map_location=device)
    vit_model.load_state_dict(state_dict)

    # Prepare for inference
    vit_model.eval()
    vit_model = vit_model.to(device)

    return vit_model, device

# Load the model and device
model, device = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess the image
            image = Image.open(filepath).convert('RGB')
            image_tensor = vit_101_transform(image).unsqueeze(0)
            
            # Ensure the tensor is on the same device as the model
            image_tensor = image_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class_idx].item() * 100
            
            # Get predicted class label
            pred_class = class_labels[pred_class_idx]
            
            # Check if the predicted item is a food item
            if not is_food_item(pred_class, confidence):
                return jsonify({
                    'error': 'The image does not appear to contain food. Please upload an image of food.',
                    'class': pred_class.replace('_', ' ').title(),
                    'confidence': round(confidence, 2),
                    'is_food': False,
                    'image_path': f'/static/uploads/{filename}'
                })
            
            # Get all nutrition information for the predicted class (for all weight options)
            nutrition_options = []
            if any(nutrition_df['label'] == pred_class):
                food_nutrition_rows = nutrition_df[nutrition_df['label'] == pred_class]
                for _, row in food_nutrition_rows.iterrows():
                    nutrition_options.append(row.to_dict())
            
            result = {
                'class': pred_class.replace('_', ' ').title(),
                'confidence': round(confidence, 2),
                'is_food': True,
                'nutrition_options': nutrition_options,
                'image_path': f'/static/uploads/{filename}'
            }
            
            return jsonify(result)
        except Exception as e:
            # Add error handling
            print(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Prediction error: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True) 
