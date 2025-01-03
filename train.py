#train.py

# Imports
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a Neural Network with PyTorch")
parser.add_argument('data_dir', type=str, help='Path to dataset')
parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in classifier')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

# Load and preprocess data
train_dir = os.path.join(args.data_dir, 'train')
valid_dir = os.path.join(args.data_dir, 'valid')

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Load pretrained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    input_size = 512

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)

if args.arch == 'vgg16':
    model.classifier = classifier
else:
    model.fc = classifier

model = model.to(device)

# Define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)

# Train the model
print("Training started...")
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validate the model
    model.eval()
    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            
            _, preds = outputs.topk(1, dim=1)
            accuracy += (preds == labels.view(*preds.shape)).sum().item()
    
    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train loss: {running_loss/len(trainloader):.3f}.. "
          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
          f"Validation accuracy: {accuracy/len(validloader.dataset):.3f}")

# Save checkpoint
checkpoint = {
    'arch': args.arch,
    'hidden_units': args.hidden_units,
    'learning_rate': args.learning_rate,
    'epochs': args.epochs,
    'state_dict': model.state_dict(),
    'class_to_idx': train_data.class_to_idx
}
torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
print("Checkpoint saved successfully!")

#correct