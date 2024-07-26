"""
Training script
Select the parameter and side to be trained by modifying PARAM and SIDE.

Run this from terminal to start Tensorboard:
    tensorboard --logdir=runs
Note: (maybe) Safari won't open the page, use chrome/firefox etc. if it doesn't work!

last change 24. Jul 2024 by Christian
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Models import ArmModel3D, ArmModel, HandModel
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import glob
import random
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

# Which arm should be trained?
ARM = 'right'  # 'left' or 'right'
PARAM = 'dir'  # 'stretch' or 'dir' or 'hand'

class PoseGestureDataset(Dataset):
    def __init__(self, data, num_landmarks, labels):
        self.labels = labels
        self.num_landmarks = num_landmarks

        self.X = []
        self.targets = {}

        for row in data.itertuples(index=False):
            landmarks = []
            for i in range(len(self.labels), num_landmarks + len(self.labels)):
                x, y, z = map(float, row[i].split(','))
                landmarks.extend([x, y, z])
            self.X.append(landmarks)

        self.X = torch.tensor(self.X, dtype=torch.float32)

        for label in self.labels:
            self.targets[label] = torch.tensor(data[label].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        sample_labels = [self.targets[label][idx] for label in self.labels]

        return sample, torch.tensor(sample_labels)

class Trainer:
    def __init__(self, param, side, config):
        self.param = param
        self.side = side

        if self.param == "dir":
            self.labels = ["az", "el"]
            self.model_file = ArmModel3D
        elif self.param == "stretch":
            self.labels = ["gesture"]
            self.model_file = ArmModel
        elif self.param == "hand":
            self.labels = ["gesture"]
            self.model_file = HandModel
        else:
            raise ValueError(f"param needs to be 'dir', 'stretch' or 'hand', not '{self.param}'!")

        self.set_hprams(config)
        self.__init_tb()
        self.__init_models()
        self.load_data()

    def set_hprams(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]

    def __init_tb(self):
        if self.param == 'hand':
            self.writer = SummaryWriter(log_dir=f'runs/hand_{time.asctime()}')
        else:
            self.writer = SummaryWriter(log_dir=f'runs/{self.param}_{self.side}_{time.asctime()}')

        self.last_loss = None

    def __init_models(self):
        self.landmark_mask = self.model_file.landmarkMask(self.side)
        self.num_landmarks = len(self.landmark_mask)
        self.model = self.model_file.GestureModel(in_feat=self.num_landmarks)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self):
        if self.param == 'hand':
            # data_directory = f'/Users/oliverparvini/Desktop/Uni/SMT/Smart-Media-Technology/TrainData/hand_data/'
            # data_directory = f'/Users/evantanggo/PycharmProjects/Smart-Media-Technology_1/TrainData/hand_data/'
            data_directory = f'/Users/christian/Desktop/UNI/SMT/HandSpreadDetection/TrainData/hand_data/'
        else:
            #data_directory = f'/Users/oliverparvini/Desktop/Uni/SMT/Smart-Media-Technology/TrainData/{self.param}_data_{self.side}'
            #data_directory = f'/Users/evantanggo/PycharmProjects/Smart-Media-Technology_1/TrainData/{self.param}_data_{self.side}'
            data_directory = f'/Users/christian/Desktop/UNI/SMT/HandSpreadDetection/TrainData/{self.param}_data_{self.side}'

        print(f"Checking directory: {data_directory}")  # Debugging statement

        if not os.path.exists(data_directory):
            print(f"Directory {data_directory} does not exist.")
            os.makedirs(data_directory)
            raise ValueError(f"Required directory {data_directory} does not exist. Please check the data directory and try again.")

        self.csv_files = glob.glob(os.path.join(data_directory, '*.csv'))
        if not self.csv_files:
            raise ValueError(f"No CSV files found in directory {data_directory}.")

        print(f"Found {len(self.csv_files)} files in {data_directory}!")
        for f in self.csv_files:
            print(f)

        data_frames = [pd.read_csv(file) for file in self.csv_files]
        if not data_frames:
            raise ValueError("No objects to concatenate. Check the data directory for CSV files.")

        data = pd.concat(data_frames, ignore_index=True)
        data_columns = [str(col_idx) for col_idx in self.landmark_mask]
        rel_columns = self.labels + data_columns
        data = data[rel_columns]

        augmented_data_noise = self.augment_data(data, augment_type='noise')
        augmented_data_scale = self.augment_data(data, augment_type='scale')

        combined_data = pd.concat([data, augmented_data_noise, augmented_data_scale])
        self.num_samples_combined = len(combined_data)

        train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

        train_dataset = PoseGestureDataset(train_data, self.num_landmarks, self.labels)
        test_dataset = PoseGestureDataset(test_data, self.num_landmarks, self.labels)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.num_samples = train_dataset.__len__()

        #self.compare_augmentation(data, augmented_data_noise, augmented_data_scale)

    def compare_augmentation(self, original_data, augmented_data_noise, augmented_data_scale):
        print("\nComparing original and augmented data...\n")
        for i in range(5):
            idx = random.randint(0, len(original_data) - 1)
            original_row = original_data.iloc[idx]
            augmented_row_noise = augmented_data_noise.iloc[idx]
            augmented_row_scale = augmented_data_scale.iloc[idx]
            print(f"Sample {i+1}:")
            print("Original landmarks:", original_row.values[1:])
            print("Augmented with noise landmarks:", augmented_row_noise.values[1:])
            print("Augmented with scale landmarks:", augmented_row_scale.values[1:])
            print()

    def loop(self, config, checkpoint_dir=None):
        if checkpoint_dir:
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "checkpoint.pth")))

        for epoch in range(config["num_epochs"]):
            epoch_targets_train = torch.tensor([])
            epoch_targets_test = torch.tensor([])
            epoch_pred_test = torch.tensor([])

            self.model.train()
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            self.writer.add_scalar('Loss/Train', loss.item(), epoch)

            self.model.eval()
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    epoch_pred_test = torch.cat((epoch_pred_test, outputs), 0)
                    epoch_targets_test = torch.cat((epoch_targets_test, targets), 0)

            self.writer.add_scalar('Loss/Test', loss.item(), epoch)

            if checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint.pth")
                torch.save(self.model.state_dict(), path)

        self.last_loss = loss.item()

    def augment_data(self, data, augment_type='noise'):
        augmented_data = []

        data_np = data.to_numpy()
        for row in data_np:

            if self.param == "dir":
                augmented_row = [row[0], row[1]]
            else:
                augmented_row = [row[0]]

            for idx in range(len(self.labels), len(row)):
                coords = np.array(row[idx].split(',')).astype(float)
                if augment_type == 'noise':
                    coords += np.random.normal(0, 0.01, coords.shape)
                elif augment_type == 'scale':
                    scale = np.random.uniform(0.9, 1.1)
                    coords *= scale
                augmented_row.append(','.join(map(str, coords)))
            augmented_data.append(augmented_row)

        augmented_df = pd.DataFrame(augmented_data, columns=data.columns)
        return augmented_df

    def save_model(self):
        # Save the trained model
        if self.param == "hand":
            save_path = f'Models/{self.param}_model.pth'
        else:
            save_path = f'Models/{self.param}_model_{self.side}.pth'

        torch.save(self.model.state_dict(), save_path)
        print("Model training completed and saved.")

    def log_hparams(self):
        self.writer.add_hparams({'epochs': self.num_epochs, 'lr': self.learning_rate, 'batchsize': self.batch_size},
                           {'loss': self.last_loss})

def train_func(config):
    myTrainer = Trainer(PARAM, ARM, config)
    myTrainer.loop(config)

    return {"loss": myTrainer.last_loss}

def save_func(config):
    myTrainer = Trainer(PARAM, ARM, config)
    myTrainer.loop(config)
    myTrainer.save_model()

if __name__ == "__main__":
    config = {
        "num_epochs": tune.grid_search([50, 100, 150]),
        "learning_rate": tune.grid_search([0.001, 0.0001]),
        "batch_size": tune.grid_search([16, 32, 64])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=1,
        reduction_factor=2
    )

    try:
        result = tune.run(
            train_func,
            resources_per_trial={"cpu": 2},
            config=config,
            num_samples=1,
            scheduler=scheduler
        )

        print("Best config: ", result.get_best_config(metric="loss", mode="min"))
        best_conf = result.get_best_config(metric="loss", mode="min")

    except Exception as e:
        print(f"An error occurred: {e}")

    save_func(best_conf)