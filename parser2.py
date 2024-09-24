import pandas as pd
from pathlib import Path
from os import path as pt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import datetime


def readCSV(path: Path = None, log=[]):
    if path:
        try:
            table = pd.read_csv(path)
            log.append(f"{datetime.datetime.now()}: Successfully added the CSV: {path.name}")
            return table, log, None
        except Exception as e:
            log.append(f"{datetime.datetime.now()}: Error finding the CSV: {path.name}, Exception: {e}")
            return None, log, None
    else:
        try:
            path = Path(input("Insert CSV path: "))
            table = pd.read_csv(path)
            log.append(f"{datetime.datetime.now()}: Successfully added the CSV: {path.name}")
            return table, log, path
        except Exception as e:
            log.append(f"{datetime.datetime.now()}: Error finding the CSV: {path.name}, Exception: {e}")
            return None, log, None


def saveCSV(path: Path = None, log: list = [], df: pd.DataFrame = None):
    if path:
        try:
            dir = pt.dirname(path)
            dtstpath = pt.join(dir, "newDataSet.csv")
            logpath = pt.join(dir, "DataSetLog.txt")
            df.to_csv(dtstpath, index=False)
            log.append(f"{datetime.datetime.now()}: Successfully saved CSV at {dtstpath}")

            with open(logpath, 'a') as f:
                for x in log:
                    f.write(x + '\n')

            return
        except Exception as e:
            log.append(f"{datetime.datetime.now()}: Error saving the CSV: {path.name}, Exception: {e}")
            return None, log
    else:
        log.append(f"{datetime.datetime.now()}: Error saving the CSV: {path.name}")
        return None, log


def featurePrep(df: pd.DataFrame, log: list = [], useful: bool = True, name: str = "", process="all"):
    if useful:
        try:
            if process == "numeration":
                encoder = LabelEncoder()
                df[name] = encoder.fit_transform(df[name])
                log.append(f"{datetime.datetime.now()}: Feature {name} successfully numerated")
            elif process == "normalisation":
                df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
                log.append(f"{datetime.datetime.now()}: Feature {name} successfully normalised")
            elif process == "one-hot":
                df = pd.get_dummies(df, columns=[name], drop_first=False)
                log.append(f"{datetime.datetime.now()}: Feature {name} successfully one-hot encoded")
            elif process == "nothing":
                log.append(f"{datetime.datetime.now()}: Feature {name}, nothing appended")
            else:
                df[name] = df[name].map(lambda a: 0 if a == process else 1)
                log.append(f"{datetime.datetime.now()}: Feature {name} successfully highlighted")
        except Exception as e:
            log.append(f"{datetime.datetime.now()}: Feature {name}, an error occurred: {e}")
    else:
        df = df.drop(name, axis=1)
        log.append(f"{datetime.datetime.now()}: Feature {name} erased successfully")
    return df, log


def parser2(df: pd.DataFrame = None, preprocessor=None):
    print(f"Pandas version: {pd.__version__}")
    log = ["Starting"]
    print(df)

    print("--------------------")

    # Fill NaNs with 0 or handle appropriately
    df.fillna(0, inplace=True)

    df, log = featurePrep(df, log, useful=True, name="attack", process="normal")

    # Create lists for numeric and categorical features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Prepare for transformations
    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

    # Apply preprocessing
    df_transformed = preprocessor.fit_transform(df)

    # Get new DataFrame with appropriate column names
    new_column_names = (numeric_features +
                        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

    df_transformed = pd.DataFrame(df_transformed, columns=new_column_names)

    # Display the transformed DataFrame
    print(df_transformed.head())

    return df_transformed, log, preprocessor


# Example usage:
if __name__ == "__main__":
    csv_path = Path("your_file.csv")  # Replace with your actual CSV file path
    data, log, _ = readCSV(csv_path)
    if data is not None:
        processed_data, log, preprocessor = parser2(data)
        # For test data, use the same preprocessor
        test_data, log_test = readCSV(Path("your_test_file.csv"))  # Replace with your actual test CSV file path
        if test_data is not None:
            test_processed_data, log_test, _ = parser2(test_data, preprocessor)
        saveCSV(csv_path, log, processed_data)
