import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Directories for localized and delocalized images
localized_dir = 'training_data/Localized'
delocalized_dir = 'training_data/Delocalized'

# Load images and labels
def load_images_and_labels(localized_dir, delocalized_dir):
    data = []
    labels = []
    
    for filename in os.listdir(localized_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(localized_dir, filename)
            img = Image.open(img_path).convert('L')
            data.append(np.array(img))
            labels.append(0)  # Localized label

    for filename in os.listdir(delocalized_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(delocalized_dir, filename)
            img = Image.open(img_path).convert('L')
            data.append(np.array(img))
            labels.append(1)  # Delocalized label

    return np.array(data), np.array(labels)

data, labels = load_images_and_labels(localized_dir, delocalized_dir)

# Split into training, validation, and test datasets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define a custom dataset
class ParticleDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = ParticleDataset(X_train, y_train, transform=transform)
val_dataset = ParticleDataset(X_val, y_val, transform=transform)
test_dataset = ParticleDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Define the Vision Transformer model
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTClassifier, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

model = ViTClassifier()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions / total_predictions

        val_loss, val_acc = evaluate_model(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

def evaluate_model(model, val_loader, criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device).float()
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    val_loss = running_loss / len(val_loader.dataset)
    val_acc = correct_predictions / total_predictions

    return val_loss, val_acc

# Train the model
num_epochs = 1
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on the test set
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
