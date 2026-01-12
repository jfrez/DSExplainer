import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import kagglehub
from kagglehub import KaggleDatasetAdapter


def load_kaggle_dataset(file_path="KaggleV2-May-2016.csv", dataset="joniarroba/noshowappointments"):
    path = kagglehub.dataset_download(dataset)
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        file_path
    )
    return df


def preprocess_split(df, target_column="No-show", drop_columns=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay"], test_size=0.1, random_state=42):
    data = df.copy()
    data = data.drop(columns=drop_columns)
    data = data.dropna()

    ordinal_features = {}
    
    for col in data.columns:
        ordinal_features[col] = data[col].unique().tolist()

    for feature, categories in ordinal_features.items():
        le = LabelEncoder()
        le.fit(categories)
        data[feature] = le.transform(data[feature])

    target = data[target_column]
    features = data.drop(columns=[target_column])

    X = features
    y = target

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
