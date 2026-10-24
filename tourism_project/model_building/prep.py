import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #data preprocessing and pipeline
from sklearn.preprocessing import LabelEncoder #text to lable encoding

df = pd.read_csv("tourism_project/data/tourism.csv")
df.drop(columns=["Unnamed: 0","CustomerID"], inplace=True) #dropping customer id 

numerical_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
# rectifying data quality issues
df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female'}) #correcting the Gender 
df['MaritalStatus'] = df['MaritalStatus'].str.strip().replace({'Unmarried': 'Single'}) #clubbing Single and Unmarried
df.describe().T

# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split into X (features) and y (target)
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
