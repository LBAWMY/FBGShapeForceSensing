import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from CustomDataset import CustomDataset
from models.FC_Model import FullyConnectedNetwork
from models.LSTM_Model import LSTMNetwork
from models.Conv1D_Model import Conv1DNetwork
from utils.shape_utils import Curve2Shape
import numpy as np
import pandas as pd

# Assuming you have your training data prepared as tensors
# train_dataset = CustomDataset(data_dir='./data/data20230819_1.csv')
# test_dataset = CustomDataset(data_dir='./data/data20230819_2.csv')
# dataset = CustomDataset(data_dir='./data/data20230819_full.csv')
# dataset = CustomDataset(data_dir='./data/data20230824_1.csv')
dataset = CustomDataset(data_dir='./data/data20230828_12c.csv')
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
# Create DataLoader for batch processing
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
# # ------------------------------------------------------------------------------------------
# option1: Hyperparameters of Conv1DNetwork
input_size = 40
output_size = [36, 2, 36, 1, 36]
model = Conv1DNetwork(input_size, output_size)
# ------------------------------------------------------------------------------------------
# # option2: Hyperparameters of LSTMNetwork
# input_size = 40
# hidden_size = 64
# num_layers = 3
# output_size = [36, 2, 36, 1, 36]
# model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)
# # ------------------------------------------------------------------------------------------
# option3: Hyperparameters of FCNetwork
# input_size = 40
# hidden_size = 64
# num_layers = 4  # You can increase this value to have more layers
# output_size = [36, 2, 36, 1, 36]
# model = FullyConnectedNetwork(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
mse = nn.MSELoss()
bce = nn.BCELoss()
ce = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

best_metric = np.inf
best_model_path = 'best_model.pth'

# Create a SummaryWriter for TensorBoard
writer = SummaryWriter()

# Training loop
num_epochs = 1000

for epoch in range(num_epochs):
    running_cur_loss, running_force_loss, running_dir_loss, running_force_loc_loss, running_twist_loss, running_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    correct_predictions, total_samples = 0, 0
    cur_err_metric, force_err_metric, force_loc_err_metric, twist_err_metric = 0.0, 0.0, 0.0, 0.0

    for batch_idx, (strain_features, curvature_target, dir_target, force_target, force_loc_target, twist_target, pos_xyz_target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients

        curvature_pred, dir_pred, force_pred, force_loc_pred, twist_pred = model(strain_features)
        force_max_index = torch.argmax(force_pred, dim=1) / (dataset.force_loc_max_tensor - dataset.force_loc_min_tensor)
        force_pred_max = torch.norm(force_pred, dim=1)
        force_target_max = torch.norm(force_target, dim=1)

        cur_loss = mse(curvature_pred, curvature_target)
        # force_loss = mse(force_pred, force_target)
        force_loss = mse(force_pred_max, force_target_max)
        # loss3 = bce(outputs3, targets3)
        dir_loss = ce(dir_pred, dir_target)
        force_loc_loss = mse(force_loc_pred, force_loc_target)
        force_loc1_loss = mse(force_max_index, force_loc_target)
        twist_loss = mse(twist_pred, twist_target)

        loss = cur_loss + force_loss + force_loc_loss + force_loc1_loss + twist_loss # + dir_loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_cur_loss += cur_loss.item()
        running_force_loss += force_loss.item()
        running_dir_loss += dir_loss.item()
        running_force_loc_loss += force_loc_loss.item()
        running_twist_loss += twist_loss.item()
        running_loss += loss.item()

        # TODO: add the force metrics: only focused on the maximum force
        # calculate the number of correct predictions
        # predicted_labels = (outputs3 >= 0.5).long() # Use 0.5 as threshold
        # predicted_labels = torch.argmax(dir_pred, dim=1)
        # correct_predictions += (predicted_labels == dir_target).sum().item()
        # total_samples += dir_target.size(0)

        # record the metrics
        cur_norm_err = curvature_pred - curvature_target
        cur_err = torch.mean(torch.abs(cur_norm_err) * (dataset.curvature_max_tensor - dataset.curvature_min_tensor), axis=1)
        cur_err_metric += torch.sum(cur_err).item()

        # force_norm_err = force_pred - force_target
        # force_err = torch.mean(torch.abs(force_norm_err) * (dataset.force_max_tensor - dataset.force_min_tensor), axis=1)
        force_norm_err = force_pred_max - force_target_max
        force_err = torch.abs(force_norm_err) * (dataset.force_max_tensor - dataset.force_min_tensor)
        force_err_metric += torch.sum(force_err).item()

        force_loc_norm_err = force_loc_pred - force_loc_target
        force_loc_err = torch.abs(force_loc_norm_err) * (dataset.force_loc_max_tensor - dataset.force_loc_min_tensor)
        force_loc_err_metric += torch.sum(force_loc_err).item()

        twist_norm_err = twist_pred - twist_target
        twist_err = torch.mean(torch.abs(twist_norm_err) * (dataset.twist_max_tensor - dataset.twist_min_tensor), axis=1)
        twist_err_metric += torch.sum(twist_err).item()

        total_samples += curvature_target.size(0)

    train_epoch_cur_loss = running_cur_loss / len(train_loader)
    train_epoch_force_loss = running_force_loss / len(train_loader)
    # train_epoch_dir_loss = running_dir_loss / len(train_loader)
    train_epoch_force_loc_loss = running_force_loc_loss / len(train_loader)
    train_epoch_twist_loss = running_twist_loss / len(train_loader)
    train_epoch_loss = running_loss / len(train_loader)
    train_epoch_acc = correct_predictions / total_samples
    train_epoch_cur_err_metric = cur_err_metric / total_samples
    train_epoch_force_err_metric = force_err_metric / total_samples
    train_epoch_force_loc_metric = force_loc_err_metric / total_samples
    train_epoch_twist_err_metric = twist_err_metric / total_samples
    print(f"Train Epoch [{epoch + 1}/{num_epochs}] - Loss: {train_epoch_loss:.4f} - Curvature Err: {train_epoch_cur_err_metric:.5f} - Force Err: {train_epoch_force_err_metric:.5f} - Force Loc Err: {train_epoch_force_loc_metric:.5f} - Twist Err: {train_epoch_twist_err_metric:.5f}")

    # Write the loss to TensorBoard
    writer.add_scalar('Train/Loss', train_epoch_loss, epoch)
    writer.add_scalar('Train/Loss: curvature', train_epoch_cur_loss, epoch)
    writer.add_scalar('Train/Loss: force', train_epoch_force_loss, epoch)
    # writer.add_scalar('Train/Loss: direction', train_epoch_dir_loss, epoch)
    writer.add_scalar('Train/Loss: force loc', train_epoch_force_loc_loss, epoch)
    writer.add_scalar('Train/Loss: twist', train_epoch_twist_loss, epoch)
    writer.add_scalar('Train/Accuracy: curvature', train_epoch_cur_err_metric, epoch)
    writer.add_scalar('Train/Accuracy: force', train_epoch_force_err_metric, epoch)
    writer.add_scalar('Train/Accuracy: force loc', train_epoch_force_loc_metric, epoch)
    writer.add_scalar('Train/Accuracy: twist', train_epoch_twist_err_metric, epoch)

    # validation loop
    if epoch % 10 == 0:
        running_cur_loss, running_force_loss, running_dir_loss, running_force_loc_loss, running_twist_loss, running_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        correct_predictions, total_samples = 0, 0
        cur_err_metric, force_err_metric, force_loc_err_metric, twist_err_metric = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (strain_features, curvature_target, dir_target, force_target, force_loc_target, twist_target,
                        pos_xyz_target) in enumerate(test_loader):
            curvature_pred, dir_pred, force_pred, force_loc_pred, twist_pred = model(strain_features) #

            cur_loss = mse(curvature_pred, curvature_target)
            force_pred_max = torch.norm(force_pred, dim=1)
            force_target_max = torch.norm(force_target, dim=1)
            force_loss = mse(force_pred_max, force_target_max)
            # force_loss = mse(force_pred, force_target)
            # loss3 = bce(outputs3, targets3)
            dir_loss = ce(dir_pred, dir_target)
            force_loc_loss = mse(force_loc_pred, force_loc_target)
            twist_loss = mse(twist_pred, twist_target)

            loss = cur_loss + force_loss + force_loc_loss + twist_loss # + dir_loss

            running_cur_loss += cur_loss.item()
            running_force_loss += force_loss.item()
            # running_dir_loss += dir_loss.item()
            running_force_loc_loss += force_loc_loss.item()
            running_twist_loss += twist_loss.item()
            running_loss += loss.item()

            # calculate the number of correct predictions
            # predicted_labels = (outputs3 >= 0.5).long()  # Use 0.5 as threshold
            # predicted_labels = torch.argmax(dir_pred, dim=1)
            # correct_predictions += (predicted_labels == dir_target).sum().item()
            # total_samples += dir_target.size(0)

            # record the metrics
            # cur_norm_err = curvature_pred - curvature_target
            # cur_err = torch.mean(torch.abs(cur_norm_err) * (dataset.curvature_max_tensor - dataset.curvature_min_tensor), axis=1)
            # cur_err_metric += torch.sum(cur_err).item()
            # total_samples += curvature_target.size(0)

            # record the metrics
            cur_norm_err = curvature_pred - curvature_target
            cur_err = torch.mean(torch.abs(cur_norm_err) * (dataset.curvature_max_tensor - dataset.curvature_min_tensor), axis=1)
            cur_err_metric += torch.sum(cur_err).item()

            # force_norm_err = force_pred - force_target
            # force_err = torch.mean(torch.abs(force_norm_err) * (dataset.force_max_tensor - dataset.force_min_tensor), axis=1)
            force_norm_err = force_pred_max - force_target_max
            force_err = torch.abs(force_norm_err) * (dataset.force_max_tensor - dataset.force_min_tensor)
            force_err_metric += torch.sum(force_err).item()

            force_loc_norm_err = force_loc_pred - force_loc_target
            force_loc_err = torch.abs(force_loc_norm_err) * (dataset.force_loc_max_tensor - dataset.force_loc_min_tensor)
            force_loc_err_metric += torch.sum(force_loc_err).item()

            twist_norm_err = twist_pred - twist_target
            twist_err = torch.mean(torch.abs(twist_norm_err) * (dataset.twist_max_tensor - dataset.twist_min_tensor), axis=1)
            twist_err_metric += torch.sum(twist_err).item()

            total_samples += curvature_target.size(0)

        test_epoch_cur_loss = running_cur_loss / len(test_loader)
        test_epoch_force_loss = running_force_loss / len(test_loader)
        # test_epoch_dir_loss = running_dir_loss / len(test_loader)
        test_epoch_force_loc_loss = running_force_loc_loss / len(test_loader)
        test_epoch_twist_loss = running_twist_loss / len(test_loader)
        test_epoch_loss = running_loss / len(test_loader)
        test_epoch_acc = correct_predictions / total_samples
        test_epoch_cur_err_metric = cur_err_metric / total_samples
        test_epoch_force_err_metric = force_err_metric / total_samples
        test_epoch_force_loc_metric = force_loc_err_metric / total_samples
        test_epoch_twist_err_metric = twist_err_metric / total_samples
        print(f"Test Epoch [{epoch + 1}/{num_epochs}] - Loss: {test_epoch_loss:.4f} - Curvature Err: {test_epoch_cur_err_metric:.5f} - Force Err: {test_epoch_force_err_metric:.5f} - Force Loc Err: {test_epoch_force_loc_metric:.5f} - Twist Err: {test_epoch_twist_err_metric:.5f}")

        # Write the loss to TensorBoard
        writer.add_scalar('Test/Loss', test_epoch_loss, epoch)
        writer.add_scalar('Test/Loss: curvature', test_epoch_cur_loss, epoch)
        writer.add_scalar('Test/Loss: force', test_epoch_force_loss, epoch)
        # writer.add_scalar('Test/Loss: direction', test_epoch_dir_loss, epoch)
        writer.add_scalar('Test/Loss: force loc', test_epoch_force_loc_loss, epoch)
        writer.add_scalar('Test/Loss: twist', test_epoch_twist_loss, epoch)
        writer.add_scalar('Test/Accuracy: curvature', test_epoch_cur_err_metric, epoch)
        writer.add_scalar('Test/Accuracy: force', test_epoch_force_err_metric, epoch)
        writer.add_scalar('Test/Accuracy: force loc', test_epoch_force_loc_metric, epoch)
        writer.add_scalar('Test/Accuracy: twist', test_epoch_twist_err_metric, epoch)

        if test_epoch_cur_err_metric < best_metric:
            best_metric = test_epoch_cur_err_metric
            # Save the best model using the SummaryWriter's log_dir
            best_model_path = f"{writer.log_dir}/best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved as {best_model_path} with accuracy: {best_metric:.6f}")

print("Training finished!")

# Close the SummaryWriter
writer.close()

# optional: use the best model for inference the test dataset
model.load_state_dict(torch.load(best_model_path))
model.eval()  # Set the model to evaluation mode

merged_results = []
with torch.no_grad():
    for batch_idx, (strain_features, curvature_target, dir_target, force_target, force_loc_target, twist_target,
                    pos_xyz_target) in enumerate(test_loader):
        curvature_pred, dir_pred, force_pred, force_loc_pred, twist_pred = model(strain_features)  #

        # de-normalization
        strain_features_array = strain_features.cpu().numpy()
        strain_features_dn_array = strain_features_array * (dataset.strain.max(axis=0) - dataset.strain.min(axis=0)) + dataset.strain.min(axis=0)
        curvature_target_array = curvature_target.cpu().numpy()
        curvature_target_dn_array = curvature_target_array * (dataset.curvature.max(axis=0) - dataset.curvature.min(axis=0)) + dataset.curvature.min(axis=0)
        dir_target_array = torch.unsqueeze(dir_target,dim=1).cpu().numpy()
        force_target_array = force_target.cpu().numpy()
        force_target_dn_array = force_target_array * (dataset.force.max(axis=0) - dataset.force.min(axis=0)) + dataset.force.min(axis=0)
        force_loc_target_array = force_loc_target.cpu().numpy()
        force_loc_target_dn_array = force_loc_target_array * (dataset.force_loc_int.max(axis=0) - dataset.force_loc_int.min(axis=0)) + dataset.force_loc_int.min(axis=0)
        twist_target_array = twist_target.cpu().numpy()
        twist_target_dn_array = twist_target_array * (dataset.twist.max(axis=0) - dataset.twist.min(axis=0)) + dataset.twist.min(axis=0)
        pos_xyz_target_array = pos_xyz_target.cpu().numpy()
        pos_xyz_target_dn_array = pos_xyz_target_array * (dataset.pos_xyz.max(axis=0) - dataset.pos_xyz.min(axis=0)) + dataset.pos_xyz.min(axis=0)
        # shape: 40 + 36 + 1 + 36 + 1 +  36 + 37*3 = 261
        raw_data = np.concatenate((strain_features_dn_array, curvature_target_dn_array, dir_target_array, force_target_dn_array, force_loc_target_dn_array, twist_target_dn_array, pos_xyz_target_dn_array), axis=1)

        curvature_pred_arr = curvature_pred.cpu().numpy()
        curvature_pred_dn_arr = curvature_pred_arr * (dataset.curvature.max(axis=0) - dataset.curvature.min(axis=0)) + dataset.curvature.min(axis=0)
        predicted_dir = torch.argmax(dir_pred, dim=1, keepdim=True).cpu().numpy()
        force_pred_arr = force_pred.cpu().numpy()
        force_pred_dn_arr = force_pred_arr * (dataset.force.max(axis=0) - dataset.force.min(axis=0)) + dataset.force.min(axis=0)
        force_loc_pred_arr = force_loc_pred.cpu().numpy()
        force_loc_pred_dn_arr = force_loc_pred_arr * (dataset.force_loc_int.max(axis=0) - dataset.force_loc_int.min(axis=0)) + dataset.force_loc_int.min(axis=0)
        twist_pred_arr = twist_pred.cpu().numpy()
        twist_pred_dn_arr = twist_pred_arr * (dataset.twist.max(axis=0) - dataset.twist.min(axis=0)) + dataset.twist.min(axis=0)
        # acquire the pos_xyz with the Curve2Shape function
        pos_xyz_pred_dn_arr = Curve2Shape(curvature_pred_dn_arr, twist_pred_dn_arr)
        # pos_xyz_pred_dn_arr = Curve2Shape(curvature_pred_dn_arr, twist_target_dn_array)
        pos_xyz_shap_error = np.abs(pos_xyz_pred_dn_arr - pos_xyz_target_dn_array) # (batch_size, 111)
        pos_x_shape_error = pos_xyz_shap_error[:, 0:37] # (batch_size, 37)

        # pos_yz_shape_error = np.sqrt(pos_xyz_shap_error[:, 37:74]**2 + pos_xyz_shap_error[:, 74:111]**2) # (batch_size, 37)
        pos_yz_norm_pred = np.sqrt(pos_xyz_pred_dn_arr[:, 37:74] ** 2 + pos_xyz_pred_dn_arr[:, 74:111] ** 2)
        pos_yz_norm_gt = np.sqrt(pos_xyz_target_dn_array[:, 37:74] ** 2 + pos_xyz_target_dn_array[:, 74:111] ** 2)
        pos_yz_shape_error = np.abs(pos_yz_norm_pred - pos_yz_norm_gt)
        pos_xyz_tip_error = np.concatenate((pos_xyz_shap_error[:, 36:37], pos_yz_shape_error[:, 36:37]), axis=1)
        # shape: 36 + 1 + 36 + 1 + 36 + 37*3 + 37 + 37 + 2 = 297
        prediction = np.concatenate((curvature_pred_dn_arr, predicted_dir, force_pred_dn_arr, force_loc_pred_dn_arr, twist_pred_dn_arr, pos_xyz_pred_dn_arr, pos_x_shape_error, pos_yz_shape_error, pos_xyz_tip_error), axis=1)
        merged_result = np.concatenate((raw_data, prediction), axis=1)

        merged_results.extend(merged_result)

# convert predictions to a numpy array
merged_results_array = np.array(merged_results)

# create a DataFrame and save a csv file
df = pd.DataFrame(merged_results_array)
csv_file_path = f"{writer.log_dir}/predictions.csv"
df.to_csv(csv_file_path, index=False)

# print basic information
print('Overall tip error: \n', np.mean(pos_xyz_tip_error, axis=0))
