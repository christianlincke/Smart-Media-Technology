import pandas as pd
import glob
import os
import csv

# Path to the directory containing the old CSV files
old_data_directory = 'hand_data/'
new_data_directory = 'hand_data/'

# Create new data directory if it doesn't exist
if not os.path.exists(new_data_directory):
    os.makedirs(new_data_directory)

# Get a list of all old CSV files in the directory
csv_files = glob.glob(old_data_directory + 'hand_gesture_data_*.csv')

# Number of landmarks
num_landmarks = 21

for old_file in csv_files:
    # Read the old CSV file
    data = pd.read_csv(old_file)

    # Prepare the new data structure
    new_data = []

    for _, row in data.iterrows():
        gesture = row['gesture']
        new_row = [gesture]
        for i in range(num_landmarks):
            x = row[f'x{i}']
            y = row[f'y{i}']
            z = row[f'z{i}']
            new_row.append(f'{x},{y},{z}')
        new_data.append(new_row)

    # Get the base name of the file to save the new file with the same name in the new directory
    base_name = os.path.basename(old_file)
    new_file = os.path.join(new_data_directory, base_name)

    # Write the new CSV file
    with open(new_file, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ['gesture'] + [f'[{i}]' for i in range(num_landmarks)]
        writer.writerow(header)
        writer.writerows(new_data)

    print(f"Converted {old_file} to {new_file}")

print("Conversion complete!")
