#predict.py

# Imports
import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

def parse_args():
    """Define command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description="Predict image classes with a trained model")
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file for category names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    """Load the model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    arch = checkpoint['arch']
    
    # Load pretrained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
    
    # Rebuild the classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_size, checkpoint['hidden_units']),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(checkpoint['hidden_units'], 102),
        torch.nn.LogSoftmax(dim=1)
    )
    
    if arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """Process an image into a format suitable for the model."""
    image = Image.open(image_path)
    
    transforms_ = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transforms_(image).unsqueeze(0)  # Add batch dimension

def predict(image_path, model, top_k, device):
    """Predict the top K classes for the given image."""
    model.to(device)
    model.eval()
    
    image = process_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities, indices = torch.exp(outputs).topk(top_k)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices.cpu().numpy()[0]]
    probabilities = probabilities.cpu().numpy()[0]
    
    return probabilities, classes

def load_category_names(json_path):
    """Load category names from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

# Script Execution
args = parse_args()
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load checkpoint and model
model = load_checkpoint(args.checkpoint)

# Predict classes
probs, classes = predict(args.image_path, model, args.top_k, device)

# Load category names if provided
if args.category_names:
    cat_to_name = load_category_names(args.category_names)
    class_names = [cat_to_name[cls] for cls in classes]
else:
    class_names = classes

# Print results
print("Predicted Classes and Probabilities:")
for i in range(args.top_k):
    print(f"{class_names[i]}: {probs[i]:.3f}")


#correct