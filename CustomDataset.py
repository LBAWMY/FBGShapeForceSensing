import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data = pd.read_csv(data_dir, header=None)
        self.strain = self.data.iloc[:, :40].values  # Strain-Shape: (num_samples, 40)
        self.curvature = self.data.iloc[:, 40:76].values  # Curvature-Shape: (num_samples, 36)
        self.dir = self.data.iloc[:, 76].values  # Direction-Shape: (num_samples, 1)
        self.force = self.data.iloc[:, 77:113].values  # Force-Shape: (num_samples, 36)
        self.force_loc_int = self.data.iloc[:, 113:114].values  # Force-Location-Shape: (num_samples, 1)
        self.force_loc_float = self.data.iloc[:, 114:115].values  # Force-Location-Shape: (num_samples, 1)
        self.twist = self.data.iloc[:, 115:151].values # Twist-Shape: (num_samples, 36)
        self.pos_xyz = self.data.iloc[:, 151:].values # PositionXYZ-Shape: (num_samples, 37+37+37)

        # normalize the input data
        self.eps = 1e-12
        self.strain_norm = (self.strain - self.strain.mean(axis=0)) / (self.strain.std(axis=0) + self.eps)
        # normalize the output data
        self.curvature_norm = (self.curvature - self.curvature.min(axis=0)) / (self.curvature.max(axis=0) - self.curvature.min(axis=0) + self.eps)
        self.dir[self.dir == -1.0] = 0
        self.dir[self.dir == 0.] = 0
        self.dir[self.dir == 1.0] = 1
        self.force_norm = (self.force - self.force.min(axis=0)) / (self.force.max(axis=0) - self.force.min(axis=0) + self.eps)
        self.force_loc_norm = (self.force_loc_int - self.force_loc_int.min(axis=0)) / (self.force_loc_int.max(axis=0) - self.force_loc_int.min(axis=0) + self.eps)
        self.twist_norm = (self.twist - self.twist.min(axis=0)) / (self.twist.max(axis=0) - self.twist.min(axis=0) + self.eps)
        self.pos_xyz_norm = (self.pos_xyz - self.pos_xyz.min(axis=0)) / (self.pos_xyz.max(axis=0) - self.pos_xyz.min(axis=0) + self.eps)
        # self.curvature_norm = (self.curvature - self.curvature.mean(axis=0)) / (self.curvature.std(axis=0) + self.eps)
        # self.dir[self.dir == -1.0] = 0
        # self.dir[self.dir == 0.] = 0
        # self.dir[self.dir == 1.0] = 1
        # self.force_norm = (self.force - self.force.mean(axis=0)) / (self.force.std(axis=0) + self.eps)
        # self.force_loc_norm = (self.force_loc_int - self.force_loc_int.mean(axis=0)) / (self.force_loc_int.std(axis=0) + self.eps)
        # self.twist_norm = (self.twist - self.twist.mean(axis=0)) / (self.twist.std(axis=0) + self.eps)
        # self.pos_xyz_norm = (self.pos_xyz - self.pos_xyz.mean(axis=0)) / (self.pos_xyz.std(axis=0) + self.eps)

        self.curvature_min_tensor = torch.tensor(self.curvature.min(axis=0), dtype=torch.float32)
        self.curvature_max_tensor = torch.tensor(self.curvature.max(axis=0), dtype=torch.float32)
        self.force_min_tensor = torch.tensor(self.force.min(axis=0), dtype=torch.float32)
        self.force_max_tensor = torch.tensor(self.force.max(axis=0), dtype=torch.float32)
        self.force_loc_min_tensor = torch.tensor(self.force_loc_int.min(axis=0), dtype=torch.float32)
        self.force_loc_max_tensor = torch.tensor(self.force_loc_int.max(axis=0), dtype=torch.float32)
        self.twist_min_tensor = torch.tensor(self.twist.min(axis=0), dtype=torch.float32)
        self.twist_max_tensor = torch.tensor(self.twist.max(axis=0), dtype=torch.float32)
        self.pos_xyz_min_tensor = torch.tensor(self.pos_xyz.min(axis=0), dtype=torch.float32)
        self.pos_xyz_max_tensor = torch.tensor(self.pos_xyz.max(axis=0), dtype=torch.float32)

        # save the statistics of the input data
        data_info_name = data_dir.replace(".csv", "_info.npz")
        np.savez(data_info_name, strain_mean=self.strain.mean(axis=0), strain_std=self.strain.std(axis=0),
                 curvature_min=self.curvature.min(axis=0), curvature_max=self.curvature.max(axis=0),
                 force_min=self.force.min(axis=0), force_max=self.force.max(axis=0),
                 force_loc_min=self.force_loc_int.min(axis=0), force_loc_max=self.force_loc_int.max(axis=0),
                 twist_min=self.twist.min(axis=0), twist_max=self.twist.max(axis=0),
                 pos_xyz_min=self.pos_xyz.min(axis=0), pos_xyz_max=self.pos_xyz.max(axis=0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        strain_features = torch.tensor(self.strain_norm[idx], dtype=torch.float32)
        curvature_target = torch.tensor(self.curvature_norm[idx], dtype=torch.float32)
        dir_target = torch.tensor(self.dir[idx], dtype=torch.long) # torch.long, torch.float32
        force_target = torch.tensor(self.force_norm[idx], dtype=torch.float32)
        force_loc_target = torch.tensor(self.force_loc_norm[idx], dtype=torch.float32)
        twist_target = torch.tensor(self.twist_norm[idx], dtype=torch.float32)
        pos_xyz_target = torch.tensor(self.pos_xyz_norm[idx], dtype=torch.float32)
        return strain_features, curvature_target, dir_target, force_target, force_loc_target, twist_target, pos_xyz_target
