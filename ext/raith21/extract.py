import os
import logging
import pandas as pd
from sim.metrics import Metrics


def ensure_output_dir(path="analysis_data"):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def extract_metrics(sim, output_dir="analysis_data"):
    """Extract all metrics dataframes and save to CSV files"""
    ensure_output_dir(output_dir)

    # Define all metrics to extract
    metrics_list = [
        "invocations",
        "scale",
        "schedule",
        "replica_deployment",
        "function_deployments",
        "function_deployment",
        "function_deployment_lifecycle",
        "functions",
        "flow",
        "network",
        "utilization",
        "fets",
    ]

    dfs = {}
    # Extract each metric
    for metric in metrics_list:
        try:
            df = sim.env.metrics.extract_dataframe(metric)
            if df is not None and not df.empty:
                # Save to CSV
                csv_path = os.path.join(output_dir, f"{metric}_df.csv")
                df.to_csv(csv_path, index=False)
                dfs[f"{metric}_df"] = df
                logging.info(f"Saved {metric} data to {csv_path}")
            else:
                logging.warning(f"No data available for {metric}")
        except Exception as e:
            logging.error(f"Error extracting {metric}: {e}")

    return dfs


# Add to your main simulation script:
def run_simulation_with_metrics():
    # After your simulation runs
    dfs = extract_metrics(sim)
    return dfs
