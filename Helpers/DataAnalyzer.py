"""
Perform Histogram Analysis of collected Data.
Useful to evaluate biases, correctness of labels etc.

"""
import pandas as pd
import glob
import os
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import time

# Create writer for logging
#writer = SummaryWriter(log_dir=f'runs/arms_3D_{ARM}_{time.asctime()}')

# Path to the directory containing the CSV files
path = f'../TrainData/'
dir_list = os.listdir(path)

print('Available directories: \n')

idx = 1
for dir in dir_list:
    print(f'[{idx}] {dir}')
    idx += 1

dir = int(input('\nWhich do you want to analyze?'))
print('->', dir_list[dir-1] + '\n')

# get all the available files
csv_files = glob.glob(path + dir_list[dir-1] + '/*.csv')

if len(list(csv_files)) == 0:
    print('Its empty!')
    quit()

print('These files are available:\n')

for file in list(csv_files):
    print(file)

print('\nAnalyze data...\n')
# Read and concatenate all CSV files into a single DataFrame
data_frames = [pd.read_csv(file) for file in csv_files]
data = pd.concat(data_frames, ignore_index=True)
data_array = data.to_numpy()

labels = np.array([])
if 'dir' in csv_files[0]:
    print('dir!')
    for row in data_array:
        labels = np.append(labels, str([row[0], row[1]])) # direction data hast 2 labels!
else:
    for row in data_array:
        labels = np.append(labels, [row[0]])

table = PrettyTable()
bins, counts = np.unique(labels, return_counts=True)

table.field_names = bins
table.add_row(counts)
print(table)

count_min, count_max = min(counts), max(counts)

plt.bar(bins, counts)
plt.ylim(count_min - (count_min / 100), count_max + (count_min / 100))
plt.title(dir_list[dir-1] + ' ' + time.strftime("%A %B %d, %Y %I:%M%p"))
plt.show()

"""
# log a histogram of all targets/predictions
writer.add_histogram('Targets/Test', epoch_targets_test, epoch)
writer.add_histogram('Predictions/Test', epoch_pred_test, epoch)

writer.flush()
writer.close()
"""