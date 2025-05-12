import logging
import os
import pandas as pd
import numpy as np

import examples.inter.main as inter
from examples.custom_function_sim.main import CustomSimulatorFactory
from sim.faassim import Simulation

logger = logging.getLogger(__name__)

def ensure_output_dir(path='analysis_data'):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def analyze_dataframe(name, df, output_dir):
    """Analyze a dataframe and print information about it"""
    print(f"\n{'='*80}")
    print(f"DataFrame: {name}")
    print(f"{'='*80}")
    
    if df is None or df.empty:
        print(f"  No data available in {name}")
        return
    
    # Print shape
    print(f"Shape: {df.shape} (rows, columns)")
    
    # Print column names and types
    print("\nColumns and Data Types:")
    for col in df.columns:
        col_type = df[col].dtype
        non_null = df[col].count()
        print(f"  {col}: {col_type} ({non_null} non-null values)")
    
    # Print sample data (first 5 rows)
    print("\nSample Data (first 5 rows):")
    print(df.head(5).to_string())
    
    # Additional stats for numeric columns
    print("\nNumeric Columns Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    else:
        print("  No numeric columns available")
    
    # For time-series data, print first and last timestamp
    time_cols = [col for col in df.columns if 'time' in col.lower() or 't_' in col.lower()]
    if time_cols:
        print("\nTime Range:")
        for col in time_cols:
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                print(f"  {col}: {df[col].min()} to {df[col].max()}")
    
    # Save DataFrame to CSV for later use
    csv_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDataFrame saved to: {csv_path}")

def main():
    logging.basicConfig(level=logging.INFO)
    output_dir = ensure_output_dir()
    
    # prepare simulation with topology and benchmark from basic example
    sim = Simulation(inter.example_topology(), inter.EnhancedBenchmark())

    # override the SimulatorFactory factory
    sim.create_simulator_factory = inter.CitySimulatorFactory

    # run the simulation
    sim.run()

    dfs = {
        'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
        'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
        'scale_df': sim.env.metrics.extract_dataframe('scale'),
        'schedule_df': sim.env.metrics.extract_dataframe('schedule'),
        'replica_deployment_df': sim.env.metrics.extract_dataframe('replica_deployment'),
        'function_deployments_df': sim.env.metrics.extract_dataframe('function_deployments'),
        'function_deployment_df': sim.env.metrics.extract_dataframe('function_deployment'),
        'function_deployment_lifecycle_df': sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        'functions_df': sim.env.metrics.extract_dataframe('functions'),
        'flow_df': sim.env.metrics.extract_dataframe('flow'),
        'network_df': sim.env.metrics.extract_dataframe('network'),
        'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
        'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets')
    }

    # Print information about each DataFrame
    for name, df in dfs.items():
        analyze_dataframe(name, df, output_dir)
        
    # Print summary information
    print("\nSummary of available DataFrames:")
    for name, df in dfs.items():
        if df is not None and not df.empty:
            print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  {name}: Empty")
    
    # Basic analysis example
    if not dfs['invocations_df'].empty:
        logger.info('Mean exec time: %.2f ms', dfs['invocations_df']['t_exec'].mean())
    
    print(f"\nAll data exported to: {os.path.abspath(output_dir)}")
    print("After reviewing the data, you can generate visualizations.")

if __name__ == '__main__':
    main()