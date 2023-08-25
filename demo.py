import torch
import torch.nn as nn
from models.LSTM_Model import LSTMNetwork
from models.FC_Model import FullyConnectedNetwork
from models.Conv1D_Model import Conv1DNetwork
import numpy as np

# Initialize the model
# # ------------------------------------------------------------------------------------------
# option1: Hyperparameters of Conv1DNetwork
# input_size = 40
# output_size = [36, 2, 36, 1, 36]
# model = Conv1DNetwork(input_size, output_size)
# ------------------------------------------------------------------------------------------
# # option2: Hyperparameters of LSTMNetwork
# input_size = 40
# hidden_size = 64
# num_layers = 3
# output_size = [36, 2, 36, 1, 36]
# model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)
# # ------------------------------------------------------------------------------------------
# option3: Hyperparameters of FCNetwork
input_size = 40
hidden_size = 64
num_layers = 4  # You can increase this value to have more layers
output_size = [36, 2, 36, 1, 36]
model = FullyConnectedNetwork(input_size, hidden_size, num_layers, output_size)

# Load the best model
best_model_path = "runs/Aug25_20-42-27_lbawmy/best_model.pt"
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set the model to evaluation mode

# Example input data for inference
input_data = np.random.rand(1, 40)  # Replace with your own input data

# 1. Preprocess the input data
# Need to load the min, max, std, and mean of previous data
data_info = np.load("data/data20230824_1_info.npz")
eps = 1e-12
input_data_norm = (input_data - data_info['strain_mean']) / (data_info['strain_std'] + eps)

# 2. Convert input data to a PyTorch tensor
strain_features_tensor = torch.tensor(input_data_norm, dtype=torch.float32)

# Perform inference
with torch.no_grad():
    curvature_pred, dir_pred, force_pred, force_loc_pred, twist_pred = model(strain_features_tensor)
    # predicted_dir = torch.argmax(dir_pred, dim=1) # depressed
    curvature_pred_arr = curvature_pred.cpu().numpy()
    curvature_pred_dn_arr = curvature_pred_arr * (data_info['curvature_max'] - data_info['curvature_min']) + data_info['curvature_min']
    predicted_dir = torch.argmax(dir_pred, dim=1, keepdim=True).cpu().numpy()
    force_pred_arr = force_pred.cpu().numpy()
    force_pred_dn_arr = force_pred_arr * (data_info['force_max'] - data_info['force_min']) + data_info['force_min']
    force_loc_pred_arr = force_loc_pred.cpu().numpy()
    force_loc_pred_dn_arr = force_loc_pred_arr * (data_info['force_loc_max'] - data_info['force_loc_min']) + data_info['force_loc_min']
    twist_pred_arr = twist_pred.cpu().numpy()
    twist_pred_dn_arr = twist_pred_arr * (data_info['twist_max'] - data_info['twist_min']) + data_info['twist_min']

print("Predicted Curvature:", curvature_pred_dn_arr)
print("Predicted Force:", force_pred_dn_arr)
print("Predicted Force Location:", force_loc_pred_dn_arr)
print("Predicted Twist:", twist_pred_dn_arr)
# print("Predicted Class:", predicted_dir)