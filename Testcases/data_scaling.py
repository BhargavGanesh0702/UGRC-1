import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

n_input_fea = 9
n_out_fea = 1

input_file = "prml_data.csv"
df = pd.read_csv(input_file)

if df.shape[1] != (n_input_fea + n_out_fea):
    raise ValueError(
        f"Total features didn't match"
    )

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_data = np.round(scaled_data.astype(np.float32), 5)
new_columns = [f"X{i+1}" for i in range(n_input_fea)] + [f"Y{i+1}" for i in range(n_out_fea)]
scaled_df = pd.DataFrame(scaled_data, columns=new_columns)

train_df, test_df = train_test_split(scaled_df, test_size=0.2, random_state=42, shuffle=True)

train_output = "scaled_prml_train.csv"
test_output = "scaled_prml_test.csv"
train_df.to_csv(train_output, index=False, float_format="%.5f")
test_df.to_csv(test_output, index=False, float_format="%.5f")
print(f"Training data saved to {train_output}")
print(f"Test data saved to {test_output}")
print("\nPreview of the scaled training data:")
print(train_df.head())