"""
Global Configuration Settings for Monetary Policy Transmission Platform

Centralized configuration for paths, parameters, and settings used across modules.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

from pathlib import Path

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
STATIONARITY_RESULTS = RESULTS_DIR / "stationarity"
ARDL_RESULTS = RESULTS_DIR / "ardl"
VAR_RESULTS = RESULTS_DIR / "var"
SIMULATION_RESULTS = RESULTS_DIR / "simulations"

# Documentation
DOCS_DIR = PROJECT_ROOT / "docs"

# =============================================================================
# DATA SETTINGS
# =============================================================================

# Default data file
DEFAULT_RAW_DATA_FILE = "nigeria_macro_data.csv"
DEFAULT_PROCESSED_DATA_FILE = "processed_macro_data.csv"

# Variable names (standardized)
VARIABLE_NAMES = {
    'mpr': 'MPR',
    'inflation': 'Inflation',
    'exchange_rate': 'ExchangeRate',
    'money_supply_m2': 'M2'
}

# VAR ordering (Cholesky decomposition)
VAR_ORDERING = ['MPR', 'ExchangeRate', 'M2', 'Inflation']

# Data period
DATA_START_DATE = "2010-01"
DATA_END_DATE = "2025-01"
DATA_FREQUENCY = "MS"  # Month start

# =============================================================================
# ECONOMETRIC PARAMETERS
# =============================================================================

# Stationarity tests
STATIONARITY_TESTS = {
    'adf': {
        'regression': 'ct',  # constant and trend
        'autolag': 'AIC',
        'maxlag': 12
    },
    'pp': {
        'regression': 'ct',
        'lags': 12
    },
    'kpss': {
        'regression': 'ct',
        'nlags': 'auto'
    }
}

# ARDL parameters
ARDL_CONFIG = {
    'max_lags': 12,
    'ic': 'aic',  # Information criterion: 'aic', 'bic', 'hqic'
    'trend': 'ct',  # constant and trend
    'bounds_test_significance': 0.05
}

# VAR parameters
VAR_CONFIG = {
    'max_lags': 12,
    'ic': 'aic',  # 'aic', 'bic', 'hqic', 'fpe'
    'trend': 'c',  # constant only (no deterministic trend in VAR)
    'verbose': False
}

# IRF parameters
IRF_CONFIG = {
    'periods': 24,  # Forecast horizon (24 months)
    'orthogonalized': True,  # Cholesky decomposition
    'impulse': None,  # All variables
    'response': None,  # All variables
    'cumulative': False,
    'bootstrap_iterations': 500,
    'confidence_level': 0.90  # 90% confidence interval
}

# FEVD parameters
FEVD_CONFIG = {
    'periods': 24,
    'orthogonalized': True
}

# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

# Policy shock scenarios
POLICY_SHOCKS = {
    'baseline': {
        'description': '+100bps (1%) MPR increase',
        'shock_size': 1.0,  # percentage points
        'shock_variable': 'MPR',
        'periods': 24
    },
    'aggressive': {
        'description': '+200bps (2%) MPR increase',
        'shock_size': 2.0,
        'shock_variable': 'MPR',
        'periods': 24
    }
}

# =============================================================================
# PLOTTING SETTINGS
# =============================================================================

PLOT_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'figsize': (14, 6),
    'dpi': 100,
    'colors': {
        'MPR': '#2E86AB',
        'Inflation': '#A23B72',
        'ExchangeRate': '#F18F01',
        'M2': '#6A994E'
    },
    'save_format': 'png',
    'save_dpi': 300
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Data quality thresholds
VALIDATION_CONFIG = {
    'min_observations': 120,  # Minimum required observations
    'max_missing_pct': 5.0,  # Maximum % missing values allowed
    'outlier_threshold': 3.0,  # Standard deviations for outlier detection
    'expected_ranges': {
        'MPR': (0, 50),
        'Inflation': (-5, 100),
        'ExchangeRate': (1, 2000),
        'M2': (0, 1e10)
    }
}

# =============================================================================
# API SETTINGS (for Spring Boot integration - Week 5)
# =============================================================================

API_CONFIG = {
    'host': 'localhost',
    'port': 8080,
    'base_url': '/api/v1',
    'endpoints': {
        'data': '/data',
        'models': '/models',
        'results': '/results',
        'simulations': '/simulations'
    }
}

# =============================================================================
# LOGGING SETTINGS
# =============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# =============================================================================
# ACADEMIC REFERENCES
# =============================================================================

CITATIONS = {
    'var_methodology': 'Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 1-48.',
    'ardl_bounds_test': 'Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. Journal of Applied Econometrics, 16(3), 289-326.',
    'identification': 'Christiano, L. J., Eichenbaum, M., & Evans, C. L. (1999). Monetary policy shocks: What have we learned and to what end? Handbook of Macroeconomics, 1, 65-148.',
    'data_source': 'Central Bank of Nigeria (2024). Statistical Bulletin. Abuja: CBN.'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        RESULTS_DIR, STATIONARITY_RESULTS, ARDL_RESULTS,
        VAR_RESULTS, SIMULATION_RESULTS, DOCS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print("✓ All directories verified/created")


def get_data_path(filename=None, processed=False):
    """
    Get path to data file.

    Args:
        filename: Name of file (default: uses config default)
        processed: If True, looks in processed/ directory

    Returns:
        Path object
    """
    if filename is None:
        filename = DEFAULT_PROCESSED_DATA_FILE if processed else DEFAULT_RAW_DATA_FILE

    directory = PROCESSED_DATA_DIR if processed else RAW_DATA_DIR
    return directory / filename


def get_results_path(model_type, filename):
    """
    Get path to results file.

    Args:
        model_type: 'stationarity', 'ardl', 'var', or 'simulations'
        filename: Name of results file

    Returns:
        Path object
    """
    results_dirs = {
        'stationarity': STATIONARITY_RESULTS,
        'ardl': ARDL_RESULTS,
        'var': VAR_RESULTS,
        'simulations': SIMULATION_RESULTS
    }

    if model_type not in results_dirs:
        raise ValueError(f"Invalid model_type: {model_type}")

    return results_dirs[model_type] / filename


# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MONETARY POLICY TRANSMISSION PLATFORM - CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nVAR Ordering: {' → '.join(VAR_ORDERING)}")
    print(f"Sample Period: {DATA_START_DATE} to {DATA_END_DATE}")
    print("\n" + "=" * 70)

    # Ensure directories exist
    ensure_directories()
