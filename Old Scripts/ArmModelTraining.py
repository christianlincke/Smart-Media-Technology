import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from Models import ArmModel
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import glob

# Which arm should be trained?
ARM = 'right'  # 'left' or 'right'
PARAM = 'stretch'  # 'direction' or 'stretch' -- later there might be elevation / '360' as well --

if PARAM != 'stretch' and PARAM != 'direction':
    raise ValueError("PARAM needs to be 'stretch' or 'direction'")

if ARM != 'left' and ARM != 'right':
    raise ValueError("ARM needs to be 'left' or 'right'")

#### HYPERPARAMETERS ###
num_epochs = 100
learning_rate = 0.001
batch_size = 32

# Create writer for logging
writer = SummaryWriter(log_dir=f'runs/arms_{PARAM}_{ARM}_{time.asctime()}')

# Custom Dataset class for PyTorch
class PoseGestureDataset(Dataset):
    def __init__(self, data):
        self.X = []
        for row in data.itertuples(index=False):
            landmarks = []
            for i in range(1, num_landmarks + 1):
                x, y, z = map(float, row[i].split(','))
                landmarks.extend([x, y, z])
            self.X.append(landmarks)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(data['gesture'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Path to the directory containing the CSV files
data_directory = f'TrainData/arm_{PARAM}_data_{ARM}/'

# Get a list of all CSV files in the directory
csv_files = glob.glob(data_directory + f'arm_{PARAM}_data_{ARM}_*.csv')

# Read and concatenate all CSV files into a single DataFrame
data_frames = [pd.read_csv(file) for file in csv_files]
data = pd.concat(data_frames, ignore_index=True)

# Extract relevant landmarks from concatenated files
# first, get the landmark mask and turn into list of strings for column name indexing
landmark_mask = ArmModel.landmarkMask(ARM)
data_columns = [str(col_idx) for col_idx in landmark_mask]
label = ['gesture'] # known column name for label

# we want label + landmarks, so we concatenate the column names:
rel_columns = label + data_columns

# indexing the pd Dataframe, only keeps the relevant columns
data = data[rel_columns]

# Number of landmarks to be used
num_landmarks = len(landmark_mask)

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create DataLoader for PyTorch
train_dataset = PoseGestureDataset(train_data)
test_dataset = PoseGestureDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function and optimizer
model = ArmModel.PoseGestureModel(in_feat=num_landmarks)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # reset list for target / prediction histograms every epoch
    epoch_targets_train = torch.tensor([])
    epoch_targets_test = torch.tensor([])
    epoch_pred_test = torch.tensor([])

    #TRAIN
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    writer.add_scalar('Loss/Train', loss.item(), epoch)

    #TEST
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            epoch_pred_test = torch.cat((epoch_pred_test, outputs), 0)
            epoch_targets_test = torch.cat((epoch_targets_test, targets), 0)

        writer.add_scalar('Loss/Test', loss.item(), epoch)

    # Logs every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # log a histogram of all targets/predictions
        writer.add_histogram('Targets/Test', epoch_targets_test, epoch)
        writer.add_histogram('Predictions/Test', epoch_pred_test, epoch)

# Save the trained model
save_path = f'../Models/'
torch.save(model.state_dict(), save_path + f'arm_{PARAM}_model_{ARM}.pth')
print("Model training completed and saved.")

landmarks, labels = next(iter(train_loader))
writer.add_graph(model, landmarks)
writer.flush()
writer.close()
