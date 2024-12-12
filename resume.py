from transformers import pipeline
import fitz  # PyMuPDF
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import pickle

# Load the second dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Output all unique entries in 'Category'
unique_categories = df['Category'].unique().tolist()
print("Unique entries in 'Category':")
print(unique_categories)


def clean_text(text):
    """Clean and normalize the text by removing special characters and converting to lowercase."""
    if isinstance(text, str):  # Check if the input is a string
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.strip()  # Remove leading/trailing whitespace
    return text

# Apply cleaning function to 'Category' and 'Resume' columns
df['Category'] = df['Category'].apply(clean_text)
df['Resume'] = df['Resume'].apply(clean_text)

# Display the cleaned data
print("\nCleaned 'Category' and 'Resume':")
print(df[['Category', 'Resume']].head())

# Split the data into input (X) and target (y)
X = df['Resume']
y = df['Category']

# Split the data into training (70%), validation (18%), and testing (12%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.40, random_state=42, stratify=y_temp)  # 0.40 of 0.30 is 12%

# Combine the splits into a single dictionary
data_splits = {
    'x_train': X_train,
    'x_val': X_val,
    'x_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test
}

# Export the data splits using pickle
with open('data_splits.pkl', 'wb') as f:
    pickle.dump(data_splits, f)

print("\nData splits have been exported to 'data_splits.pkl'")
