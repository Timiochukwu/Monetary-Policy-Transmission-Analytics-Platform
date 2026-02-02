"""
Data Loader for Nigerian Monetary Policy Transmission Analysis

This module loads raw macroeconomic data from CBN/NBS sources and prepares it
for econometric analysis.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional


class NigerianMacroDataLoader:
    """
    Loads and preprocesses Nigerian macroeconomic data for monetary policy
    transmission analysis.

    Variables:
    - MPR: Monetary Policy Rate (%)
    - Inflation: Headline CPI inflation (YoY %)
    - Exchange Rate: NGN/USD (NAFEX/I&E Window)
    - M2: Broad Money Supply (Naira billions)
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Root directory containing raw/ and processed/ subdirectories
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_path = self.data_dir / "metadata.json"

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)

    def load_raw_data(self, filename: str = "nigeria_macro_data.csv") -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Args:
            filename: Name of the CSV file in data/raw/

        Returns:
            DataFrame with date index and macro variables
        """
        file_path = self.raw_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please download data from CBN or use template data."
            )

        # Load CSV
        df = pd.read_csv(file_path)

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Sort by date
        df.sort_index(inplace=True)

        print(f"✓ Loaded {len(df)} observations from {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform basic data validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'n_observations': len(df),
            'date_range': (df.index[0], df.index[-1]),
            'missing_values': df.isnull().sum().to_dict(),
            'columns': df.columns.tolist(),
            'warnings': []
        }

        # Check minimum observations
        if len(df) < 120:
            validation_results['warnings'].append(
                f"Only {len(df)} observations available. Minimum 120 recommended for VAR."
            )

        # Check for missing values
        if df.isnull().sum().sum() > 0:
            validation_results['warnings'].append(
                "Missing values detected. These will need imputation."
            )

        # Check variable ranges (basic sanity checks)
        if 'mpr' in df.columns:
            if df['mpr'].min() < 0 or df['mpr'].max() > 50:
                validation_results['warnings'].append(
                    f"MPR values outside expected range: {df['mpr'].min():.1f}% - {df['mpr'].max():.1f}%"
                )

        if 'exchange_rate' in df.columns:
            if df['exchange_rate'].min() < 1:
                validation_results['warnings'].append(
                    "Exchange rate values suspiciously low. Check data units."
                )

        return validation_results

    def create_analysis_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create analysis-ready dataset with proper column naming and ordering.

        Args:
            df: Raw DataFrame

        Returns:
            Processed DataFrame ready for econometric analysis
        """
        # Rename columns to standard format (if needed)
        column_mapping = {
            'mpr': 'MPR',
            'inflation': 'Inflation',
            'exchange_rate': 'ExchangeRate',
            'money_supply_m2': 'M2'
        }

        df_analysis = df.rename(columns=column_mapping)

        # Ensure proper ordering (matches VAR ordering: MPR -> ER -> M2 -> Inf)
        column_order = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
        df_analysis = df_analysis[column_order]

        return df_analysis

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_macro_data.csv"):
        """
        Save processed data to data/processed/ directory.

        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path)
        print(f"✓ Saved processed data to {output_path}")

    def get_data_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics summary.

        Args:
            df: DataFrame to summarize

        Returns:
            DataFrame with descriptive statistics
        """
        summary = df.describe().T
        summary['Missing'] = df.isnull().sum()
        summary['Missing %'] = (df.isnull().sum() / len(df) * 100).round(2)

        return summary

    def load_and_prepare(self) -> pd.DataFrame:
        """
        Complete pipeline: load, validate, and prepare data.

        Returns:
            Analysis-ready DataFrame
        """
        print("=" * 60)
        print("NIGERIAN MONETARY POLICY DATA LOADER")
        print("=" * 60)

        # Step 1: Load raw data
        print("\n[1/4] Loading raw data...")
        df_raw = self.load_raw_data()

        # Step 2: Validate
        print("\n[2/4] Validating data...")
        validation = self.validate_data(df_raw)
        print(f"  - Observations: {validation['n_observations']}")
        print(f"  - Period: {validation['date_range'][0].strftime('%Y-%m')} to {validation['date_range'][1].strftime('%Y-%m')}")
        print(f"  - Variables: {', '.join(validation['columns'])}")

        if validation['warnings']:
            print("\n  ⚠ Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")

        # Step 3: Create analysis dataset
        print("\n[3/4] Creating analysis dataset...")
        df_analysis = self.create_analysis_dataset(df_raw)

        # Step 4: Summary statistics
        print("\n[4/4] Summary statistics:")
        summary = self.get_data_summary(df_analysis)
        print(summary.to_string())

        # Save processed data
        self.save_processed_data(df_analysis)

        print("\n" + "=" * 60)
        print("✓ DATA LOADING COMPLETE")
        print("=" * 60)

        return df_analysis


def main():
    """
    Main execution: Load and prepare Nigerian macroeconomic data.
    """
    # Initialize loader
    loader = NigerianMacroDataLoader(data_dir="data")

    # Load and prepare data
    df = loader.load_and_prepare()

    return df


if __name__ == "__main__":
    df = main()
