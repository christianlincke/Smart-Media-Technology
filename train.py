"""
Training script, improved
- now using a class!

Run this from terminal to start Tensorboard:
    tensorboard --logdir=runs
Note: Safari won't open the page, you need chrome/firefox...!

last change 9. Jul 2024 by Christian
"""
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

# Which arm should be trained?
ARM = 'right'  # 'left' or 'right'
PARAM = 'stretch' # 'stretch' or 'dir' or 'hand'

# Custom Dataset class for PyTorch
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

    def __init__(self, param, side):
        """
        initialize Class
        :param param: str "hand", "stretch" or "dir"
        :param side: str "left" or "right"
        """

        self.param = param
        self.side = side

        # assign models and labels depending on param to be trained
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

        # init hyperparameters
        self.set_hprams()

        # init tensorboard
        self.__init_tb()

        # init models
        self.__init_models()

        # load data
        self.load_data()


    def set_hprams(self):
        """
        define hyperparameters
        :return:
        """
        self.num_epochs = 100
        self.learning_rate = 0.001
        self.batch_size = 32

    def __init_tb(self):
        """
        init tensorboard
        :return:
        """
        # Create writer for logging
        if self.param == 'hand':
            self.writer = SummaryWriter(log_dir=f'runs/hand_{time.asctime()}')
        else:
            self.writer = SummaryWriter(log_dir=f'runs/{self.param}_{self.side}_{time.asctime()}')

        # variable to store last loss values
        self.last_loss = None

    def __init_models(self):
        """
        init model to be trained
        :return:
        """

        # first, get the landmark mask and turn into list of strings for column name indexing
        self.landmark_mask = self.model_file.landmarkMask(self.side)
        # Number of landmarks to be used
        self.num_landmarks = len(self.landmark_mask)

        # Initialize the model, loss function and optimizer
        self.model = self.model_file.GestureModel(in_feat=self.num_landmarks)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_data(self):
        """
        load all recorded data for the selected param and side
        :return:
        """
        # Path to the directory containing the CSV files
        if self.param == 'hand':
            data_directory = f'TrainData/hand_data/'
        else:
            data_directory = f'TrainData/{self.param}_data_{self.side}/'

        # Get a list of all CSV files in the directory
        self.csv_files = glob.glob(data_directory + f'*.csv')
        print(f"Found {len(list(self.csv_files))} files!")
        for f in list(self.csv_files):
            print(f)

        # Read and concatenate all CSV files into a single DataFrame
        data_frames = [pd.read_csv(file) for file in self.csv_files]
        data = pd.concat(data_frames, ignore_index=True)

        # Extract relevant landmarks from concatenated files
        data_columns = [str(col_idx) for col_idx in self.landmark_mask]

        # we want label + landmarks, so we concatenate the column names:
        rel_columns = self.labels + data_columns

        # indexing the pd Dataframe, only keeps the relevant columns
        data = data[rel_columns]

        # Augment the data
        augmented_data_noise = self.augment_data(data, augment_type='noise')
        augmented_data_scale = self.augment_data(data, augment_type='scale')

        # Combine original and augmented data
        combined_data = pd.concat([data, augmented_data_noise, augmented_data_scale])
        self.num_samples_combined = len(combined_data) #

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42)

        # Create DataLoader for PyTorch
        train_dataset = PoseGestureDataset(train_data, self.num_landmarks, self.labels)
        test_dataset = PoseGestureDataset(test_data, self.num_landmarks, self.labels)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.num_samples = train_dataset.__len__()

        # Debug: Überprüfe einige Augmentationsbeispiele
        self.compare_augmentation(data, augmented_data_noise, augmented_data_scale)

    def compare_augmentation(self, original_data, augmented_data_noise, augmented_data_scale):
        """
        Compare original and augmented data samples.
        """
        print("\nComparing original and augmented data...\n")
        for i in range(5):  # compare 5 random samples
            idx = random.randint(0, len(original_data) - 1)
            original_row = original_data.iloc[idx]
            augmented_row_noise = augmented_data_noise.iloc[idx]
            augmented_row_scale = augmented_data_scale.iloc[idx]
            print(f"Sample {i+1}:")
            print("Original landmarks:", original_row.values[1:])
            print("Augmented with noise landmarks:", augmented_row_noise.values[1:])
            print("Augmented with scale landmarks:", augmented_row_scale.values[1:])
            print()

    def loop(self):
        """
        run the train and test loops
        :return:
        """
        # Training loop
        for epoch in range(self.num_epochs):
            # reset list for target / prediction histograms every epoch
            epoch_targets_train = torch.tensor([])
            epoch_targets_test = torch.tensor([])
            epoch_pred_test = torch.tensor([])

            # TRAIN
            self.model.train()
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            self.writer.add_scalar('Loss/Train', loss.item(), epoch)

            # TEST
            self.model.eval()
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                    epoch_pred_test = torch.cat((epoch_pred_test, outputs), 0)
                    epoch_targets_test = torch.cat((epoch_targets_test, targets), 0)

                self.writer.add_scalar('Loss/Test', loss.item(), epoch)

            # Logs every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

                # log a histogram of all targets/predictions
                self.writer.add_histogram('Targets/Test', epoch_targets_test, epoch)
                self.writer.add_histogram('Predictions/Test', epoch_pred_test, epoch)

        # log last loss value
        self.last_loss = loss.item()

    @staticmethod
    def augment_data(data, augment_type='noise'):
        """
        Perform data augmentation on the landmark data.
        Data is stored as landmarks (x,y,z), so we need to implement custom augmentations.

        :param data: pd.DataFrame dataset to be augmented
        :param augment_type: str type of augmentation ('noise' or 'scale')
        :return: augmented dataset
        """
        augmented_data = []

        data_np = data.to_numpy()
        for row in data_np:
            augmented_row = [row[0]]  # include the gesture label
            for idx in range(1, len(row)):
                coords = np.array(row[idx].split(',')).astype(float)
                if augment_type == 'noise':
                    # Apply random noise
                    coords += np.random.normal(0, 0.01, coords.shape)
                elif augment_type == 'scale':
                    # Apply random scaling
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

    def log(self):
        landmarks, labels = next(iter(self.train_loader))
        self.writer.add_graph(self.model, landmarks)

        self.writer.add_hparams({"training files": str(self.csv_files),
                                 "num samples raw": self.num_samples,
                                 "num samples incl. aug": self.num_samples_combined,
                                 "epochs": self.num_epochs,
                                 "batch size": self.batch_size,
                                 "learning rate": self.learning_rate},
                                {"loss": self.last_loss})
        self.writer.flush()
        self.writer.close()

if __name__ == "__main__":
    myTrainer = Trainer(PARAM, ARM)
    myTrainer.loop()
    myTrainer.save_model()
    myTrainer.log()
