import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import datasets, transforms
from torch.optim import Adam
import os

# Paths
train_dir = './resized_apples/'  # Directory containing resized_badapples and resized_goodapples
pretrained_model_path = 'pytorch_model.bin'  # Path to your saved model weights
results_dir = './fine_tuned_model/'  # Directory to save the fine-tuned model

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the feature extractor (for pre-processing)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Load the pre-trained ViT model for fine-tuning
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=2)

# Load the state_dict but force loading onto the CPU
state_dict = torch.load(pretrained_model_path, map_location=torch.device('cpu'))

# Remove the classifier weights from the state_dict since they have a size mismatch
del state_dict['classifier.weight']
del state_dict['classifier.bias']

# Load the rest of the state_dict into the model
model.load_state_dict(state_dict, strict=False)

# Move the model to the appropriate device (GPU/CPU)
model = model.to(device)

# Data Augmentation and Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 as expected by ViT
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load the dataset
full_dataset = datasets.ImageFolder(train_dir, transform=transform)

# Split dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-5)  # Use Adam optimizer with a low learning rate
criterion = nn.CrossEntropyLoss()  # Since this is a binary classification

# Early Stopping Parameters
patience = 3  # Stop training if no improvement after 3 epochs
best_val_accuracy = 0.0
epochs_without_improvement = 0

# Function to evaluate the model on validation data
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation during validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Training Loop with Early Stopping
num_epochs = 20  # Train for a maximum number of epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # Training step
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs.logits, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Calculate and print training loss and accuracy
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Evaluate the model on validation data after each epoch
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Early Stopping Check
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0  # Reset the counter
        # Save the model if it has the best performance so far
        os.makedirs(results_dir, exist_ok=True)
        model_save_path = os.path.join(results_dir, 'fine_tuned_vit_model_best.bin')
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved at {model_save_path}")
    else:
        epochs_without_improvement += 1
        print(f"Validation accuracy did not improve for {epochs_without_improvement} epoch(s)")

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered. Stopping training after {epoch + 1} epochs.")
        break

# Final model saving (in case training completes without early stopping)
model_save_path = os.path.join(results_dir, 'fine_tuned_vit_model_final.bin')
torch.save(model.state_dict(), model_save_path)
print(f"Training complete. Final model saved at {model_save_path}")
