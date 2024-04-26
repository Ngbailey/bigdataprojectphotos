import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt


def construct_graph(features):
    # Calculate adjacency matrix based on spatial relationships (e.g., 8-connected neighbors)
    # For simplicity, let's assume a grid-like structure for pixels
    image_size = int(np.sqrt(features.shape[0] / 3))  # Assuming RGB images
    adj_matrix = np.zeros((image_size * image_size, image_size * image_size))
    for i in range(image_size):
        for j in range(image_size):
            index = i * image_size + j
            if i > 0:
                adj_matrix[index, index - image_size] = 1  # Upper neighbor
                if j > 0:
                    adj_matrix[index, index - image_size - 1] = 1  # Upper-left neighbor
                if j < image_size - 1:
                    adj_matrix[index, index - image_size + 1] = 1  # Upper-right neighbor
            if j > 0:
                adj_matrix[index, index - 1] = 1  # Left neighbor
            if j < image_size - 1:
                adj_matrix[index, index + 1] = 1  # Right neighbor
            if i < image_size - 1:
                adj_matrix[index, index + image_size] = 1  # Lower neighbor
                if j > 0:
                    adj_matrix[index, index + image_size - 1] = 1  # Lower-left neighbor
                if j < image_size - 1:
                    adj_matrix[index, index + image_size + 1] = 1  # Lower-right neighbor

    # Convert adjacency matrix to edge index
    edge_index = np.transpose(np.nonzero(adj_matrix))

    # Convert edge_index to PyTorch tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create a Data object from node features and edge index
    data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index)

    return data



# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.mean(x, dim=0)  # Pooling to get a graph-level representation

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Load datasets
train_dataset = datasets.ImageFolder('seg_train/seg_train', transform=transform)
test_dataset = datasets.ImageFolder('seg_test/seg_test', transform=transform)

# Define DataLoader for train and test sets
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define a simple feature extractor (fully connected layer)
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image tensor
        x = self.fc(x)
        return x

# Instantiate the feature extractor
image_dim = 150 * 150 * 3  # Flatten image size (150x150x3)
feature_dim = 64  # Dimensionality of the extracted features
feature_extractor = FeatureExtractor(image_dim, feature_dim)

# Instantiate the GCN model
input_dim = feature_dim  # Use the output dimension of the feature extractor as input to GCN
hidden_dim = 64
output_dim = 6  # Number of categories
model = GCN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # Define number of epochs
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        # Extract features from images using the feature extractor
        features = feature_extractor(data)
        # Construct graph (for simplicity, not implemented here)
        graph = construct_graph(features)# Construct graph from features
        output = model(graph)
        loss = criterion(output.unsqueeze(0), labels)
        loss.backward()
        optimizer.step()

# Evaluation on test set
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for data, labels in test_loader:
        features = feature_extractor(data)
        graph = construct_graph(features) # Construct graph from features
        output = model(graph)
        pred = output.argmax(dim=0)
        total_correct += (pred == labels).sum().item()
        total_samples += len(labels)

accuracy = total_correct / total_samples
print("Accuracy on test set:", accuracy)


# Assuming `output` contains the output of the GCN model
output_data = output.detach().numpy()  # Convert PyTorch tensor to NumPy array

# Check the shape and contents of the output_data array
print("Shape of output_data:", output_data.shape)
print("Contents of output_data:", output_data)

# Visualize the first two dimensions of the node features (assuming output_data has shape [num_nodes, feature_dim])
if len(output_data.shape) >= 2:  # Check if there are at least two dimensions
    plt.scatter(output_data[:, 0], output_data[:, 1])  # Scatter plot of the first two dimensions
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Node Features Visualization')
    plt.show()
else:
    print("Node features have fewer than two dimensions, unable to visualize.")

# Load the prediction dataset with a custom folder structure
pred_dataset = datasets.ImageFolder('seg_pred', transform=transform)

# Define DataLoader for the prediction dataset
pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)

# Make predictions
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for data, labels in pred_loader:
        # Preprocess images (if necessary)
        # Extract features using feature extractor (if applicable)
        features = feature_extractor(data)  # Assuming you have a feature extractor

        # Construct graph
        graph = construct_graph(features)

        # Make prediction
        output = model(graph)
        
        # Ensure the output tensor has the expected shape
        if len(output.shape) == 1:  # If the output tensor is 1-dimensional
            pred = output.argmax().item()  # Apply argmax directly
        else:
            pred = output.argmax(dim=-1).item()  # Apply argmax along the last dimension
        
        predictions.append(pred)
        true_labels.append(labels.item())  # Convert label tensor to scalar and append to true_labels

# Map predictions to corresponding class labels
try:
    predicted_classes = [pred_dataset.classes[pred] for pred in predictions]
    print("Predicted classes:", predicted_classes)
except IndexError:
    print("Predicted indices are out of range:", predictions)
    print("Number of classes:", len(pred_dataset.classes))

# Compare predicted classes with true labels
correct_predictions = sum(pred == true_label for pred, true_label in zip(predictions, true_labels))
total_predictions = len(pred_dataset)
accuracy = correct_predictions / total_predictions
print("Accuracy:", accuracy)

