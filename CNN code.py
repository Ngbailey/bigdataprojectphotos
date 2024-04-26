import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

# Load a pre-trained CNN model (e.g., ResNet50)
model = models.resnet50(pretrained=True)
# Freeze the weights of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to fit the input size of the pre-trained model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])

# Directory containing your images
image_dir = r"C:\Users\natha\Desktop\bigdataprojectphotos\seg_train\seg_train\buildings"


# List to store extracted features
extracted_features = []

print("Image directory:", image_dir)

# Iterate over all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load and preprocess the image
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

        # Pass the image through the pre-trained model to extract features
        with torch.no_grad():
            features = model(image_tensor)

        # Flatten the feature tensor and append to the list
        flattened_features = torch.flatten(features, start_dim=1)
        extracted_features.append(flattened_features)

# Stack the extracted features into a single tensor
extracted_features_tensor = torch.stack(extracted_features)

# Print the shape of the extracted features tensor
print("Shape of extracted features:", extracted_features_tensor.shape)

