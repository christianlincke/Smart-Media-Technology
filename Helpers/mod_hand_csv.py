import pandas as pd
import os


def rename_columns_in_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Rename the columns
    new_columns = ['gesture'] + [str(i) for i in range(21)]
    df.columns = new_columns

    # Save the modified DataFrame back to CSV
    df.to_csv(file_path, index=False)
    print(f"Renamed columns in {file_path}")


# Specify the directory containing the CSV files
directory = '../TrainData/hand_data/'

# Loop through all CSV files in the directory and rename columns
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        rename_columns_in_csv(file_path)

