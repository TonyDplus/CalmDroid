import numpy as np
import pandas as pd


def add_noise_to_last_six_columns(df, noise_ratio):
    # Copy the original dataframe to avoid modifying it directly
    df_noisy = df.copy()

    # Get the last six columns
    last_six_columns = df_noisy.columns[-6:]

    # Calculate the number of rows to add noise to, based on the noise_ratio
    num_rows = df_noisy.shape[0]
    num_noisy_rows = int(num_rows * noise_ratio)

    # Randomly select rows for noise addition
    rows_to_modify = np.random.choice(df_noisy.index, size=num_noisy_rows, replace=False)

    # Apply noise: flip 0 to 1 and 1 to 0 in the selected rows and last six columns
    for column in last_six_columns:
        df_noisy.loc[rows_to_modify, column] = 1 - df_noisy.loc[rows_to_modify, column]

    return df_noisy


# File path and read the original data
file_path = 'data/180sample/VS_train.csv'
df = pd.read_csv(file_path)

# Apply noise with an example ratio (e.g., 4%)
df_noisy = add_noise_to_last_six_columns(df, noise_ratio=0.005)

# Save the noisy dataframe to a new CSV file
df_noisy.to_csv('VS_noise/VS_train_noisy_5.csv', index=False)
