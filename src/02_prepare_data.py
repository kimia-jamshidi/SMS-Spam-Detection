
import pandas as pd
from pathlib import Path

from text_preprocessing import preprocess_message


# Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "sms-spam-collection.csv"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed_sms.csv"


# Load raw dataset
df = pd.read_csv(RAW_DATA_PATH)

# Apply shared text preprocessing
df["message_final"] = (
    df["message"]
    .fillna("")
    .astype(str)
    .apply(preprocess_message)
)

# Save processed dataset
df.to_csv(PROCESSED_DATA_PATH, index=False)

print("Data preprocessing completed successfully.")
print(f"Processed data saved to: {PROCESSED_DATA_PATH}")
print(df[["label", "message", "message_final"]].head())
