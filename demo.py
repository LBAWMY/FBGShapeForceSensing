import torch
import torch.nn as nn
from models.LSTM_Model import LSTMNetwork  # Assuming you have defined the model architecture
import numpy as np

# Initialize the model
# # ------------------------------------------------------------------------------------------
# # option1: Hyperparameters of Conv1DNetwork
# input_size = 30
# output_size = [30, 30, 1]
# model = Conv1DNetwork(input_size, output_size)
# ------------------------------------------------------------------------------------------
# option2: Hyperparameters of LSTMNetwork
input_size = 30
hidden_size = 64
num_layers = 3
output_size = [30, 30, 1]
model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)
# # ------------------------------------------------------------------------------------------
# # option3: Hyperparameters of FCNetwork
# input_size = 30
# hidden_size = 64
# num_layers = 4  # You can increase this value to have more layers
# output_size = [30, 30, 1]
# model = FullyConnectedNetwork(input_size, hidden_size, num_layers, output_size)

# Load the best model
best_model_path = "runs/Aug11_14-57-14_lbawmy/best_model.pt"
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set the model to evaluation mode

# Example input data for inference
input_data = np.random.rand(1, 30)  # Replace with your own input data

# Convert input data to a PyTorch tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32)

# Perform inference
with torch.no_grad():
    outputs1, outputs2, outputs3 = model(input_tensor)
    predicted_dir = 1 if outputs3.item() >= 0.5 else 0

print("Predicted Curvature:", outputs1.cpu().numpy())
print("Predicted Force:", outputs2.cpu().numpy())
print("Predicted Probability:", outputs3.item())
print("Predicted Class:", predicted_dir)