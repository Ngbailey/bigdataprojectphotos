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
import os
from PIL import Image
import time
import random

# Set seed for Python's built-in random module
random_seed = 1362
random.seed(random_seed)

# Set seed for NumPy
np.random.seed(random_seed)

# Set seed for PyTorch
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  

# Define a function to extract adjacency matrix from edge indices
def extract_adjacency(edge_index, num_nodes):
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    for i in range(edge_index.size(1)):
        src, dest = edge_index[0, i].item(), edge_index[1, i].item()
        adjacency[src, dest] = 1
        adjacency[dest, src] = 1  # Assuming undirected graph

    return adjacency

start_time = time.time()  # Start the timer

def construct_graph(features):
    # Calculate adjacency matrix based on spatial relationships (e.g., 8-connected neighbors)
    # Grid-like structure for pixels
    image_size = int(np.sqrt(features.shape[0] / 3))  # RGB images
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





# Define the GCN model with L2 regularization
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, weight_decay):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Define L2 regularization term
        self.weight_decay = weight_decay

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Calculate L2 regularization term
        l2_reg = self.weight_decay * sum(param.pow(2).sum() for param in self.parameters())

        return torch.mean(x, dim=0) - 0.5 * l2_reg  # Add L2 regularization term to the output


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

# Instantiate the GCN model with L2 regularization
input_dim = feature_dim  # Use the output dimension of the feature extractor as input to GCN
hidden_dim = 64
output_dim = 6  # Number of categories
weight_decay = 1e-4  # Strength of L2 regularization
model = GCN(input_dim, hidden_dim, output_dim, weight_decay)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Check data loading and feature extraction
for i in range(5):  # Print features of the first 5 images
    sample_data, sample_labels = next(iter(train_loader))
    sample_features = feature_extractor(sample_data)
    print("Sample Image Shape:", sample_data.shape)
    print("Sample Features Shape:", sample_features.shape)
    # Print the first few features
    print("Sample Features:", sample_features[0, :10])  # Print the first 10 features
    # Optionally, visualize the sample image
    plt.imshow(sample_data.squeeze().permute(1, 2, 0))
    plt.show()

# Training loop
train_outputs = [] # List to store the outputs for training data
num_epochs = 150  # Define number of epochs
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        features = feature_extractor(data)# Extract features from images using the feature extractor
        graph = construct_graph(features)# Construct graph from features
        output = model(graph)
        train_outputs.append(output.detach().numpy())
        loss = criterion(output.unsqueeze(0), labels)
        loss.backward()
        optimizer.step()







# Evaluation on test set
model.eval()
total_correct = 0
total_samples = 0
adjacency_matrices = []  # List to store adjacency matrices
with torch.no_grad():
    for data, labels in test_loader:
        features = feature_extractor(data)
        graph = construct_graph(features)  # Construct graph from features
        output = model(graph)
        pred = output.argmax(dim=0)
        total_correct += (pred == labels).sum().item()
        total_samples += len(labels)
        
        # Extract adjacency matrix
        edge_index = graph.edge_index
        num_nodes = graph.num_nodes
        adjacency_matrix = extract_adjacency(edge_index, num_nodes)
        adjacency_matrices.append(adjacency_matrix)

accuracy = total_correct / total_samples
print("Accuracy on test set:", accuracy)





output_data = output.detach().numpy()  # Convert PyTorch tensor to NumPy array

# Check the shape and contents of the output_data array
print("Shape of output_data:", output_data.shape)
print("Contents of output_data:", output_data)









# Stop the timer and calculate elapsed time just wanted to see how long it would take to do training/testing
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

# Make predictions on unlabeled data (prediction images)
prediction_path = 'pred_small'
prediction_images = []

# Iterate over the files in the prediction directory
for file_name in os.listdir(prediction_path):
    # Load the image and append it to the list of images
    image_path = os.path.join(prediction_path, file_name)
    image = Image.open(image_path).convert('RGB')
    prediction_images.append(image)

# Preprocess the prediction images
preprocessed_images = [transform(image).unsqueeze(0) for image in prediction_images]

# Make predictions on the preprocessed images
predictions = []
model.eval()
with torch.no_grad():
    for image_tensor in preprocessed_images:
        features = feature_extractor(image_tensor)
        graph = construct_graph(features)
        output = model(graph)
        pred = output.argmax().item()  # Directly get the argmax
        predictions.append(pred)

print("Predictions:", predictions)
# Count the occurrences of each predicted label
label_counts = {label: predictions.count(label) for label in set(predictions)}

# Calculate the total number of predictions
total_predictions = len(predictions)

# Define the mapping dictionary for numerical labels to class names
label_to_class = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}

# Convert numerical labels to class names in predictions
class_predictions = [label_to_class[label] for label in predictions]

# Print the predictions with class names
print("Predictions:", class_predictions)


# Print the label distribution
print("Label distribution:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} predictions, {count / total_predictions:.2%} of total")

# Calculate the most frequent predicted label
most_frequent_label = max(label_counts, key=label_counts.get)
print("Most frequent predicted label:", most_frequent_label)


# Manually created true labels to test prediction
true_labels = [5, 0, 4, 2, 5, 1, 4, 3, 3, 2, 2, 3, 0, 4, 1, 5, 3, 5, 0, 3, 1, 5, 3, 3, 1, 5, 1, 2, 0, 3, 5, 3, 5, 0, 2, 2, 4, 4, 0, 3, 0, 5, 2, 0, 0, 1, 1, 0, 4, 5, 4, 1, 3, 2, 3, 5, 1, 3, 3, 1, 0, 4, 5, 3, 2, 0, 3, 4, 4, 3, 3, 0, 0, 3, 3, 3, 5, 2, 1, 4, 0, 5, 4, 3, 0, 3, 3, 2, 3, 5, 5, 3, 3, 1, 4, 2, 3, 3, 0, 0, 3, 0, 2, 3, 3, 1, 5, 5, 3, 3, 0, 0, 1, 1, 4, 3, 2, 4, 2, 3, 2, 3, 4, 3, 3, 3, 4, 4, 1, 3, 1, 1, 3, 0, 1, 3, 5, 2, 1, 3, 2, 4, 2, 3, 1, 3, 5, 2, 2, 1, 5, 5, 3, 0, 3, 3, 4, 4, 2, 2, 4, 0, 0, 1, 2, 5, 1, 1, 3, 1, 1, 4, 4, 5, 1, 4, 2, 4, 1, 5, 1, 1, 2, 0, 3, 2, 0, 3, 0, 1, 5, 5, 0, 1, 3, 3, 3, 4, 3, 4, 3, 0, 3, 5, 5, 5, 3, 1, 1, 0, 0, 1, 3, 3, 2, 5]

# Calculate the accuracy
correct_predictions = sum(1 for pred, true_label in zip(predictions, true_labels) if pred == true_label)
total_predictions = len(true_labels)
accuracy = correct_predictions / total_predictions

print("Accuracy:", accuracy)



