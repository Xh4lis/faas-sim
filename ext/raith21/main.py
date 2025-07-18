#!/usr/bin/env python
# coding: utf-8
"""
FaaS Simulation with Raith21 AI Workloads
Main simulation runner with configurable parameters
"""

import logging
import random
import time
import os
import argparse
import sys
from tqdm import tqdm
import numpy as np

# Analysis and reporting
from examples.analysis.report import main as report_main

# Core simulation components
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex
from sim.core import Environment
from sim.docker import ContainerRegistry
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation
from sim.logging import SimulatedClock, RuntimeLogger
from sim.metrics import Metrics, PowerMetrics
from sim.skippy import SimulationClusterContext

# Raith21 components
from ext.raith21 import images
from ext.raith21.benchmark.constant import ConstantBenchmark
from ext.raith21.characterization import get_raith21_function_characterizations
from ext.raith21.deployments import create_all_deployments
from ext.raith21.etherdevices import convert_to_ether_nodes
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.functionsim import AIPythonHTTPSimulatorFactory
from ext.raith21.generator import generate_devices
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.generators.edgegpu import edgegpu_settings
from ext.raith21.generators.edgetpu import edgetpu_settings
from ext.raith21.generators.edgecloudlet import edgecloudlet_settings
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu
from ext.raith21.resources import ai_resources_per_node_image
from ext.raith21.topology import urban_sensing_topology
from ext.raith21.util import vanilla

# MHFD extensions
from ext.mhfd.deployments import create_smart_city_deployments, create_custom_smart_city_deployments
from ext.mhfd.scbenchmark import create_smart_city_constant_benchmark
from ext.mhfd.power import Raith21PowerOracle, DEVICE_POWER_PROFILES, monitor_power_consumption, power_monitoring_loop
from ext.mhfd.autoscaler import create_heterogeneous_edge_autoscaler


class SimulationConfig:
    """Configuration parameters for the simulation"""
    
    def __init__(self):
        # Simulation parameters
        self.num_devices = 5000
        self.device_settings = edgegpu_settings
        self.duration = 500
        self.total_rps = 1200
        self.scenario = "custom"  # Options: "default", "intensive", "distributed", "custom"
        
        # Custom function counts for scenario
        self.custom_counts = {
            "resnet50-inference": 8,
            "speech-inference": 7,
            "resnet50-preprocessing": 5,
            "resnet50-training": 3,  
            "python-pi": 4,
            "fio": 0,
        }
        
        # Scheduler parameters
        self.percentage_of_nodes_to_score = 100
        
        # Output configuration
        self.settings_id = "new_pwr_monitor_"
        self.data_dir_base = "./data"
        self.vis_dir_base = "./Vis"
        
        # Seeds for reproducibility
        self.numpy_seed = 1435
        self.random_seed = 1435


def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(level=logging.INFO)
    # Uncomment for debug logging
    # logging.getLogger("sim.faas.scaling").setLevel(logging.DEBUG)
    # logging.getLogger("sim.faas").setLevel(logging.DEBUG)


def set_random_seeds(config):
    """Set random seeds for reproducible results"""
    np.random.seed(config.numpy_seed)
    random.seed(config.random_seed)


def setup_devices_and_topology(config):
    """Generate devices and convert to network topology"""
    print(f"Generating {config.num_devices} devices...")
    devices = generate_devices(config.num_devices, config.device_settings)
    ether_nodes = convert_to_ether_nodes(devices)
    
    storage_index = StorageIndex()
    topology = urban_sensing_topology(ether_nodes, storage_index)
    
    return devices, ether_nodes, topology, storage_index


def setup_oracles():
    """Initialize prediction oracles"""
    fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
    resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
    power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
    
    return fet_oracle, resource_oracle, power_oracle


def setup_scheduler_config(config, fet_oracle, resource_oracle):
    """Configure scheduler predicates and priorities"""
    predicates = []
    predicates.extend(Scheduler.default_predicates)
    predicates.extend([
        # CanRunPred(fet_oracle, resource_oracle),  # Uncomment if needed
        NodeHasAcceleratorPred(),
        NodeHasFreeGpu(),
        # NodeHasFreeTpu(),  # Uncomment if needed
    ])
    
    priorities = vanilla.get_priorities()
    
    return {
        "percentage_of_nodes_to_score": config.percentage_of_nodes_to_score,
        "priorities": priorities,
        "predicates": predicates,
    }


def create_benchmark(config):
    """Create workload benchmark based on scenario"""
    if config.scenario == "custom":
        return create_smart_city_constant_benchmark(
            duration=config.duration,
            total_rps=config.total_rps,
            scenario="custom",
            custom_counts=config.custom_counts
        )
    else:
        return create_smart_city_constant_benchmark(
            duration=config.duration,
            total_rps=config.total_rps,
            scenario=config.scenario
        )


def print_topology_analysis(devices, ether_nodes, topology):
    """Print detailed topology analysis for debugging"""
    print(f"\n=== TOPOLOGY SUMMARY ===")
    print(f"Total nodes in topology: {len(topology.nodes)}")
    
    # Separate compute nodes from network infrastructure
    compute_nodes = []
    infrastructure_nodes = []
    
    for node_name, node in topology.nodes.items():
        name_str = str(node_name)
        if any(keyword in name_str.lower() for keyword in ['link', 'switch', 'shared', 'registry']):
            infrastructure_nodes.append((node_name, node))
        else:
            compute_nodes.append((node_name, node))
    
    print(f"Compute nodes: {len(compute_nodes)}")
    print(f"Infrastructure nodes: {len(infrastructure_nodes)}")
    
    # Analyze compute nodes
    print(f"\n=== COMPUTE NODES ANALYSIS ===")
    compute_node_types = {}
    
    for node_name, node in compute_nodes[:20]:  # Show first 20
        arch = getattr(node, 'arch', None)
        if arch is None:
            labels = getattr(node, 'labels', {})
            arch = labels.get('arch', 'Unknown')
        
        accelerators = getattr(node, 'accelerators', [])
        capacity = getattr(node, 'capacity', {})
        
        name_str = str(node_name)
        print(f"{name_str:<15s} - Arch: {str(arch):<10s} - Accelerators: {accelerators}")
        if capacity:
            print(f"                    Capacity: {capacity}")
        
        arch_str = str(arch)
        compute_node_types[arch_str] = compute_node_types.get(arch_str, 0) + 1
    
    print(f"\n=== COMPUTE NODE TYPE DISTRIBUTION ===")
    for arch, count in compute_node_types.items():
        print(f"{arch}: {count} nodes")
    
    # Comparison analysis
    print(f"\n=== ORIGINAL DEVICES vs TOPOLOGY NODES ===")
    print(f"Generated devices: {len(devices)}")
    print(f"Ether nodes: {len(ether_nodes)}")
    print(f"Topology compute nodes: {len(compute_nodes)}")
    
    # Sample comparisons
    print(f"\n=== SAMPLE ORIGINAL DEVICES ===")
    for i, device in enumerate(devices[:5]):
        print(f"Device {i}: {device.arch.name} - {device.accelerator.name} - {device.location.name}")
    
    print(f"\n=== SAMPLE ETHER NODES ===")
    for i, ether_node in enumerate(ether_nodes[:5]):
        arch = getattr(ether_node, 'arch', 'Unknown')
        accelerators = getattr(ether_node, 'accelerators', [])
        print(f"Ether {i}: {ether_node.name} - Arch: {arch} - Accelerators: {accelerators}")


def setup_environment(topology, storage_index, sched_params, fet_oracle, resource_oracle, power_oracle):
    """Initialize and configure the simulation environment"""
    env = Environment()
    
    # Configure core components
    env.simulator_factory = AIPythonHTTPSimulatorFactory(
        get_raith21_function_characterizations(resource_oracle, fet_oracle)
    )
    env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
    env.topology = topology
    env.faas = DefaultFaasSystem(env)
    env.container_registry = ContainerRegistry()
    env.storage_index = storage_index
    env.cluster = SimulationClusterContext(env)
    env.scheduler = Scheduler(env.cluster, **sched_params)
    
    # Power and autoscaling components
    power_metrics = PowerMetrics()
    env.power_oracle = power_oracle
    env.power_metrics = power_metrics
    
    autoscaler = create_heterogeneous_edge_autoscaler(
        env, env.faas, power_oracle, strategy="basic"
    )
    
    # Background processes
    env.background_processes.append(power_monitoring_loop)
    env.background_processes.append(lambda env: autoscaler.run())
    
    return env


def run_simulation(topology, benchmark, env):
    """Execute the simulation and return results"""
    print("Starting simulation...")
    sim = Simulation(topology, benchmark, env=env)
    result = sim.run()
    print("Simulation completed!")
    return sim


def extract_metrics(sim):
    """Extract all metrics dataframes from simulation"""
    metric_types = [
        "invocations", "scale", "schedule", "replica_deployment",
        "function_deployments", "function_deployment", "function_deployment_lifecycle",
        "functions", "flow", "network", "utilization", "fets",
        "power", "energy", "scaling_decisions", "scaling_evaluations"
    ]
    
    dfs = {}
    for metric_type in metric_types:
        dfs[f"{metric_type}_df"] = sim.env.metrics.extract_dataframe(metric_type)
    
    return dfs


def save_dataframes(dfs, data_dir):
    """Save all dataframes to CSV files"""
    os.makedirs(data_dir, exist_ok=True)
    
    for df_name, df in dfs.items():
        if df is not None and not df.empty:
            output_path = os.path.join(data_dir, f"{df_name}.csv")
            df.to_csv(output_path, index=False)
            print(f"Saved {df_name} to {output_path}")
        else:
            print(f"Skipped {df_name} (empty or None DataFrame)")


def print_dataframe_info(dfs):
    """Print summary information about extracted dataframes"""
    print(f"\nExtracted {len(dfs)} dataframes:")
    
    for df_name, df in dfs.items():
        print(f"\n{df_name} columns:")
        if df is not None and not df.empty:
            print(f"  Shape: {df.shape}")
            print("  Columns:")
            for col in df.columns:
                sample = df[col].iloc[0] if len(df) > 0 else None
                sample_type = type(sample).__name__ if sample is not None else "N/A"
                print(f"    - {col}: {sample_type}")
        else:
            print("  Empty or None DataFrame")


def generate_reports(data_dir, vis_dir):
    """Generate analysis reports from simulation data"""
    os.makedirs(vis_dir, exist_ok=True)
    
    # Store and modify sys.argv for report generator
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], "--data-dir", data_dir, "--output-dir", vis_dir]
    
    print(f"Generating analysis reports from {data_dir}/ to {vis_dir}/")
    try:
        report_main()
    finally:
        sys.argv = original_argv


def get_output_directories(config, benchmark):
    """Generate output directory names based on configuration"""
    device_id = f"d{config.num_devices}"
    rps_id = f"r{benchmark.rps}"
    
    data_dir = os.path.join(config.data_dir_base, f"{config.settings_id}_{device_id}_{rps_id}")
    vis_dir = os.path.join(config.vis_dir_base, f"{config.settings_id}_{device_id}_{rps_id}")
    
    return data_dir, vis_dir


def main():
    """Main simulation runner"""
    # Initialize configuration
    config = SimulationConfig()
    
    # Setup
    setup_logging()
    set_random_seeds(config)
    
    # Generate infrastructure
    devices, ether_nodes, topology, storage_index = setup_devices_and_topology(config)
    fet_oracle, resource_oracle, power_oracle = setup_oracles()
    
    # Print topology analysis
    print_topology_analysis(devices, ether_nodes, topology)
    
    # Configure scheduler and benchmark
    sched_params = setup_scheduler_config(config, fet_oracle, resource_oracle)
    benchmark = create_benchmark(config)
    
    # Setup and run simulation
    env = setup_environment(topology, storage_index, sched_params, fet_oracle, resource_oracle, power_oracle)
    sim = run_simulation(topology, benchmark, env)
    
    # Extract and analyze results
    dfs = extract_metrics(sim)
    print_dataframe_info(dfs)
    
    # Save results and generate reports
    data_dir, vis_dir = get_output_directories(config, benchmark)
    save_dataframes(dfs, data_dir)
    generate_reports(data_dir, vis_dir)
    
    print(f"\nSimulation complete! Results saved to {data_dir}/ and {vis_dir}/")


if __name__ == "__main__":
    main()
