
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# DATA CLEANING AND PREPROCESSING PROJECT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "raw_data.csv"
OUTPUT_DIR = BASE_DIR / "output"


# Step 1: Load the dataset
def load_data():
    print("Loading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        na_values=["", " ", "NA", "N/A", "null", "None"]
    )

    print("Dataset loaded successfully!")
    return df


# Step 2: Explore the raw data
def explore_data(df):
    print("\n========== ORIGINAL DATA ==========")
    print(df.head())

    print("\nDataset shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())


# Step 3: Clean and preprocess data
def clean_data(df):
    print("\nCleaning data...")

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove unnecessary spaces from text
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].str.strip()

    # Standardize names
    if "Name" in df.columns:
        df["Name"] = df["Name"].str.title()

    # Standardize city names
    if "City" in df.columns:
        df["City"] = df["City"].str.title()

    # Standardize gender values
    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"]
            .str.lower()
            .replace({
                "m": "male",
                "f": "female"
            })
            .str.title()
        )

    # Convert numeric columns
    for column in ["Age", "Marks"]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Replace invalid ages with missing values
    if "Age" in df.columns:
        df.loc[
            ~df["Age"].between(0, 120),
            "Age"
        ] = np.nan

    # Marks must be between 0 and 100
    if "Marks" in df.columns:
        df.loc[
            ~df["Marks"].between(0, 100),
            "Marks"
        ] = np.nan

    # Fill missing numeric values with median
    for column in ["Age", "Marks"]:
        if column in df.columns:
            median_value = df[column].median()

            if pd.notna(median_value):
                df[column] = df[column].fillna(
                    median_value
                )

    # Fill missing categorical values with mode
    for column in ["Gender", "City"]:
        if column in df.columns and df[column].isna().any():
            mode_values = df[column].mode(dropna=True)

            fill_value = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "Unknown"
            )

            df[column] = df[column].fillna(fill_value)

    # Fill missing names
    if "Name" in df.columns:
        df["Name"] = df["Name"].fillna("Unknown")

    # Standardize email addresses
    if "Email" in df.columns:
        df["Email"] = df["Email"].str.lower()
        df["Email"] = df["Email"].fillna("Unknown")

    # Correct date formats
    if "Join_Date" in df.columns:
        dates = pd.to_datetime(
            df["Join_Date"],
            errors="coerce",
            dayfirst=True
        )

        df["Join_Date"] = (
            dates.dt.strftime("%Y-%m-%d")
            .fillna("Unknown")
        )

    print("Data cleaning completed!")
    return df


# Step 4: Save cleaned data and report
def save_data(original_df, cleaned_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned CSV
    output_file = OUTPUT_DIR / "cleaned_data.csv"
    cleaned_df.to_csv(output_file, index=False)

    # Create cleaning report
    report_file = OUTPUT_DIR / "cleaning_report.txt"

    report = f"""
DATA CLEANING AND PREPROCESSING REPORT
======================================

Original rows: {len(original_df)}
Cleaned rows: {len(cleaned_df)}

Duplicate rows removed:
{len(original_df) - len(original_df.drop_duplicates())}

Missing values BEFORE cleaning:
{original_df.isnull().sum().to_string()}

Missing values AFTER cleaning:
{cleaned_df.isnull().sum().to_string()}

Cleaned dataset shape: {cleaned_df.shape}

Output file: {output_file}
"""

    report_file.write_text(report, encoding="utf-8")

    print("\nCleaned data saved to:", output_file)
    print("Report saved to:", report_file)


# Step 5: Main program
def main():
    try:
        original_df = load_data()

        explore_data(original_df)

        cleaned_df = clean_data(original_df)

        print("\n========== CLEANED DATA ==========")
        print(cleaned_df.head(10))

        print("\nMissing values after cleaning:")
        print(cleaned_df.isnull().sum())

        save_data(original_df, cleaned_df)

        print("\nPROJECT COMPLETED SUCCESSFULLY!")

    except FileNotFoundError:
        print("ERROR: raw_data.csv was not found.")
        print("Place raw_data.csv in the same folder as main.py.")

    except Exception as error:
        print("An error occurred:", error)


if __name__ == "__main__":
    main()
