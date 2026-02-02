"""
Data Validation Module for Nigerian Monetary Policy Analysis

Performs quality checks, outlier detection, and data integrity validation
for macroeconomic time series data.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings


class DataValidator:
    """
    Validates macroeconomic data quality for econometric analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with DataFrame.

        Args:
            df: DataFrame with macroeconomic variables
        """
        self.df = df
        self.validation_report = {}

    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in each variable.

        Returns:
            Dictionary with count of missing values per variable
        """
        missing_counts = self.df.isnull().sum().to_dict()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict()

        self.validation_report['missing_values'] = {
            'counts': missing_counts,
            'percentages': missing_pct
        }

        return missing_counts

    def detect_outliers(self, threshold: float = 3.0) -> Dict[str, List[Tuple]]:
        """
        Detect outliers using z-score method (|z| > threshold).

        Args:
            threshold: Number of standard deviations for outlier detection

        Returns:
            Dictionary with outliers per variable
        """
        outliers = {}

        for col in self.df.columns:
            # Calculate z-scores
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())

            # Find outliers
            outlier_indices = z_scores[z_scores > threshold].index
            outlier_values = [(idx.strftime('%Y-%m'), self.df.loc[idx, col])
                             for idx in outlier_indices]

            if outlier_values:
                outliers[col] = outlier_values

        self.validation_report['outliers'] = outliers
        return outliers

    def check_monotonicity(self, variables: List[str] = None) -> Dict[str, str]:
        """
        Check if certain variables are monotonic (e.g., M2 should generally increase).

        Args:
            variables: List of variables to check (default: ['M2'])

        Returns:
            Dictionary with monotonicity status
        """
        if variables is None:
            variables = ['M2'] if 'M2' in self.df.columns else []

        monotonicity = {}

        for var in variables:
            if var not in self.df.columns:
                continue

            diff = self.df[var].diff()
            increasing_pct = (diff > 0).sum() / len(diff) * 100
            decreasing_pct = (diff < 0).sum() / len(diff) * 100

            if increasing_pct > 80:
                monotonicity[var] = "Mostly increasing (expected for M2)"
            elif decreasing_pct > 80:
                monotonicity[var] = "Mostly decreasing (unusual for M2)"
            else:
                monotonicity[var] = f"Mixed ({increasing_pct:.1f}% increasing)"

        self.validation_report['monotonicity'] = monotonicity
        return monotonicity

    def check_value_ranges(self) -> Dict[str, Dict]:
        """
        Check if values are within expected economic ranges.

        Returns:
            Dictionary with range checks per variable
        """
        range_checks = {}

        # Expected ranges (based on Nigerian economic history)
        expected_ranges = {
            'MPR': (0, 50),  # Policy rate between 0-50%
            'Inflation': (-5, 100),  # Inflation can be negative (rare) or very high
            'ExchangeRate': (1, 2000),  # NGN/USD from inception to present
            'M2': (0, 1e10)  # Broad money (billions of Naira)
        }

        for col in self.df.columns:
            if col in expected_ranges:
                min_val, max_val = expected_ranges[col]
                actual_min = self.df[col].min()
                actual_max = self.df[col].max()

                in_range = (actual_min >= min_val) and (actual_max <= max_val)

                range_checks[col] = {
                    'expected': expected_ranges[col],
                    'actual': (actual_min, actual_max),
                    'valid': in_range
                }

        self.validation_report['range_checks'] = range_checks
        return range_checks

    def check_temporal_consistency(self) -> Dict[str, bool]:
        """
        Check that dates are properly ordered and no duplicates exist.

        Returns:
            Dictionary with temporal consistency checks
        """
        temporal_checks = {}

        # Check for duplicate dates
        has_duplicates = self.df.index.duplicated().any()
        temporal_checks['no_duplicate_dates'] = not has_duplicates

        # Check if sorted
        is_sorted = self.df.index.is_monotonic_increasing
        temporal_checks['dates_sorted'] = is_sorted

        # Check for missing months
        expected_freq = pd.infer_freq(self.df.index)
        temporal_checks['inferred_frequency'] = expected_freq if expected_freq else "Irregular"

        self.validation_report['temporal_consistency'] = temporal_checks
        return temporal_checks

    def validate_all(self) -> Dict:
        """
        Run all validation checks.

        Returns:
            Complete validation report
        """
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        # 1. Missing values
        print("\n[1/5] Checking missing values...")
        missing = self.check_missing_values()
        if sum(missing.values()) == 0:
            print("  ✓ No missing values detected")
        else:
            print("  ⚠ Missing values found:")
            for var, count in missing.items():
                if count > 0:
                    pct = self.validation_report['missing_values']['percentages'][var]
                    print(f"    - {var}: {count} ({pct}%)")

        # 2. Outliers
        print("\n[2/5] Detecting outliers (3σ threshold)...")
        outliers = self.detect_outliers()
        if not outliers:
            print("  ✓ No significant outliers detected")
        else:
            print("  ⚠ Outliers found:")
            for var, outlier_list in outliers.items():
                print(f"    - {var}: {len(outlier_list)} outliers")
                for date, value in outlier_list[:3]:  # Show first 3
                    print(f"      {date}: {value:.2f}")

        # 3. Value ranges
        print("\n[3/5] Checking value ranges...")
        ranges = self.check_value_ranges()
        for var, check in ranges.items():
            status = "✓" if check['valid'] else "⚠"
            print(f"  {status} {var}: {check['actual'][0]:.2f} to {check['actual'][1]:.2f}")

        # 4. Temporal consistency
        print("\n[4/5] Checking temporal consistency...")
        temporal = self.check_temporal_consistency()
        for check, result in temporal.items():
            status = "✓" if result or check == 'inferred_frequency' else "⚠"
            print(f"  {status} {check}: {result}")

        # 5. Monotonicity (for M2)
        print("\n[5/5] Checking monotonicity...")
        monotonic = self.check_monotonicity()
        for var, status in monotonic.items():
            print(f"  - {var}: {status}")

        print("\n" + "=" * 60)
        print("✓ VALIDATION COMPLETE")
        print("=" * 60)

        return self.validation_report

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of validation results.

        Returns:
            DataFrame with validation summary
        """
        summary_data = []

        for col in self.df.columns:
            row = {
                'Variable': col,
                'N_Obs': len(self.df[col]),
                'Missing': self.df[col].isnull().sum(),
                'Min': self.df[col].min(),
                'Max': self.df[col].max(),
                'Mean': self.df[col].mean(),
                'Std': self.df[col].std()
            }
            summary_data.append(row)

        return pd.DataFrame(summary_data)


def main():
    """
    Example usage of DataValidator.
    """
    # This would typically be imported and used with loaded data
    # Example:
    # from data_loader import NigerianMacroDataLoader
    # loader = NigerianMacroDataLoader()
    # df = loader.load_and_prepare()
    # validator = DataValidator(df)
    # validator.validate_all()

    print("DataValidator module loaded successfully.")
    print("Import this module to validate your macroeconomic data.")


if __name__ == "__main__":
    main()
