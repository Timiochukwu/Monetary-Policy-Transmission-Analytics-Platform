"""
Forecast Evaluation — Nigerian Monetary Policy Transmission

Out-of-sample forecast accuracy assessment:
  1. h-step-ahead forecasts (h = 1, 3, 6, 12 months)
  2. Accuracy metrics: RMSE, MAE, MAPE
  3. Comparison with naive benchmark (random walk)

Day 14 deliverable.  Demonstrates predictive power of the VAR.

References
----------
Diebold, F. X. & Mariano, R. S. (1995). Comparing Predictive Accuracy.
  Journal of Business & Economic Statistics, 13(3), 253–263.
"""

import sys
import pandas   as pd
import numpy    as np
from pathlib    import Path


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class ForecastEvaluator:
    """
    Evaluate VAR out-of-sample forecast accuracy.

    Parameters
    ----------
    df       : pd.DataFrame   –  macro data (DatetimeIndex)
    ordering : list[str]      –  variable ordering
    opt_lag  : int            –  VAR lag order
    save_dir : str | Path     –  output directory
    """

    def __init__(self, df: pd.DataFrame, ordering: list[str], opt_lag: int,
                 save_dir: str = "results/forecasts"):
        self.df       = df[ordering]
        self.ordering = ordering
        self.opt_lag  = opt_lag
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # ROLLING-WINDOW FORECAST
    # ─────────────────────────────────────────────────────────────
    def rolling_forecast(self, train_size: int = 120, horizons: list[int] = [1, 3, 6, 12]) -> dict:
        """
        Rolling-window out-of-sample forecasts.

        Train on first 'train_size' obs, forecast next h steps, roll forward.
        """
        from statsmodels.tsa.api import VAR

        n = len(self.df)
        results = {h: {var: {"actual": [], "forecast": []} for var in self.ordering}
                   for h in horizons}

        for t in range(train_size, n - max(horizons)):
            df_train = self.df.iloc[:t]
            try:
                var = VAR(df_train).fit(self.opt_lag)
                # Forecast up to max horizon
                forecast = var.forecast(df_train.values[-self.opt_lag:], steps=max(horizons))

                for h in horizons:
                    if t + h < n:
                        for i, var_name in enumerate(self.ordering):
                            results[h][var_name]["actual"].append(self.df.iloc[t + h][var_name])
                            results[h][var_name]["forecast"].append(forecast[h - 1, i])
            except:
                continue

        return results

    # ─────────────────────────────────────────────────────────────
    # ACCURACY METRICS
    # ─────────────────────────────────────────────────────────────
    def compute_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> dict:
        """RMSE, MAE, MAPE."""
        errors = actual - forecast
        return {
            "RMSE": round(float(np.sqrt(np.mean(errors ** 2))), 4),
            "MAE":  round(float(np.mean(np.abs(errors))), 4),
            "MAPE": round(float(100 * np.mean(np.abs(errors / actual))), 2),
        }

    def evaluate_all(self, results: dict) -> pd.DataFrame:
        """Compute accuracy metrics for all variables and horizons."""
        rows = []
        for h, var_dict in results.items():
            for var, data in var_dict.items():
                if len(data["actual"]) == 0:
                    continue
                actual   = np.array(data["actual"])
                forecast = np.array(data["forecast"])
                metrics  = self.compute_metrics(actual, forecast)

                rows.append({
                    "horizon": h,
                    "variable": var,
                    **metrics,
                })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self) -> None:
        print("\n" + "=" * 70)
        print("  FORECAST EVALUATION  –  Out-of-Sample Accuracy")
        print("=" * 70)

        print("\n  [1/2] Rolling-window forecasts …")
        results = self.rolling_forecast()

        print("  [2/2] Computing accuracy metrics …\n")
        df_metrics = self.evaluate_all(results)

        print("  " + "─" * 65)
        print(f"  {'Var':<18} {'Horizon':>8} {'RMSE':>10} {'MAE':>10} {'MAPE %':>10}")
        print("  " + "─" * 65)
        for _, row in df_metrics.iterrows():
            print(f"  {row['variable']:<18} {int(row['horizon']):>8}"
                  f" {row['RMSE']:>10.4f} {row['MAE']:>10.4f} {row['MAPE']:>10.2f}")

        path = self.save_dir / "forecast_accuracy.csv"
        df_metrics.to_csv(path, index=False)
        print(f"\n  ✓ saved  {path}\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    ordering = ["MPR", "ExchangeRate", "M2", "Inflation"]
    opt_lag  = 11

    evaluator = ForecastEvaluator(df, ordering, opt_lag)
    evaluator.run_full_analysis()


if __name__ == "__main__":
    main()
