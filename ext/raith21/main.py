#!/usr/bin/env python
# coding: utf-8
import logging
import random
import time
import os
from tqdm import tqdm
import numpy as np
import argparse
import sys
from examples.analysis.report import main as report_main
from skippy.core.scheduler import Scheduler  # Kubernetes-inspired scheduler
from skippy.core.storage import StorageIndex  # Tracks data locations in the network

# Import Raith21-specific modules
from ext.raith21 import images  # Container image definitions for AI workloads
from ext.raith21.benchmark.constant import (
    ConstantBenchmark,
)  # Defines constant rate workloads
from ext.raith21.characterization import (
    get_raith21_function_characterizations,
)  # Function behavior models
from ext.raith21.deployments import (
    create_all_deployments,
)  # Creates function deployment specs
from ext.mhfd.deployments import (
    create_smart_city_deployments,
    create_custom_smart_city_deployments
)
from ext.raith21.etherdevices import (
    convert_to_ether_nodes,
)  # Converts device specs to network nodes
from ext.raith21.fet import (
    ai_execution_time_distributions,
)  # Function execution time data
from ext.raith21.functionsim import (
    AIPythonHTTPSimulatorFactory,
)  # Function execution simulator
from ext.raith21.generator import generate_devices  # Creates heterogeneous device specs
from ext.raith21.generators.cloudcpu import (
    cloudcpu_settings,
)  # Device configuration settings
from ext.raith21.generators.edgegpu import edgegpu_settings  # Device configuration settings
from ext.raith21.generators.edgetpu import edgetpu_settings
from ext.raith21.generators.edgecloudlet import edgecloudlet_settings
from ext.raith21.generators.cloudcpu import cloudcpu_settings
from ext.raith21.oracles import (
    Raith21ResourceOracle,
    Raith21FetOracle,
)  # Performance prediction oracles
from ext.raith21.predicates import (
    CanRunPred,
    NodeHasAcceleratorPred,
    NodeHasFreeGpu,
    NodeHasFreeTpu,
)  # Node filters

from ext.raith21.resources import ai_resources_per_node_image  # Resource usage data
from ext.raith21.topology import (
    urban_sensing_topology,
)  # Edge-fog-cloud network topology
from ext.raith21.util import vanilla  # Default scheduling priorities
from sim.core import Environment  # Main simulation environment
from sim.docker import ContainerRegistry  # Simulates container image registry
from sim.faas.system import DefaultFaasSystem  # FaaS platform implementation
from sim.faassim import Simulation  # Core simulation engine
from sim.logging import SimulatedClock, RuntimeLogger  # Time and event logging
from sim.metrics import Metrics  # Performance metrics collection
from sim.skippy import SimulationClusterContext  # Cluster abstraction for scheduler

# Set seeds for reproducible simulation results
np.random.seed(1435)
random.seed(1435)
logging.basicConfig(level=logging.INFO)

# Generate heterogeneous edge and cloud devices
num_devices = 100  # Min 24 - Controls simulation scale
devices = generate_devices(num_devices, cloudcpu_settings)
ether_nodes = convert_to_ether_nodes(devices)  # Convert to network topology nodes

scenario = "custom"  # Start with default scenario


# Create oracles for predicting execution times and resource requirements
fet_oracle = Raith21FetOracle(
    ai_execution_time_distributions
)  # Function Execution Time oracle
resource_oracle = Raith21ResourceOracle(
    ai_resources_per_node_image
)  # Resource usage oracle

# Set up function deployments and container images
# deployments = list(
#     create_all_deployments(fet_oracle, resource_oracle).values()
# )  # Function deployment specs
# all_deployments = create_all_deployments(fet_oracle, resource_oracle)

# # Choose your specific function pool
# selected_functions = [
#     "resnet50-inference",     # High compute inference
#     "mobilenet-inference",    # Lightweight inference  
#     "speech-inference",       # Audio processing
#     "resnet50-training",    # Comment out heavy training
#     "resnet50-preprocessing" # Comment out preprocessing
# ]

# # Filter deployments to only include selected functions
# deployments = [all_deployments[func] for func in selected_functions if func in all_deployments]

# print("Selected functions for simulation:")
# for func_name in selected_functions:
#     if func_name in all_deployments:
#         print(f"  ✓ {func_name}")
#     else:
#         print(f"  ✗ {func_name} (not available)")

if scenario == "custom":
    # Custom instance counts for specific use case
    custom_counts = {
        "resnet50-inference": 8,      # 8 high-resolution camera zones
        "mobilenet-inference": 15,    # 15 lightweight edge zones
        "speech-inference": 6,        # 6 audio monitoring zones
        "resnet50-preprocessing": 8,  # 8 data processing zones
        "resnet50-training": 3,       # 3 training instances
    }
    deployments = create_custom_smart_city_deployments(
        fet_oracle, resource_oracle, custom_counts
    )
else:
    # Use predefined scenario
    deployments = create_smart_city_deployments(
        fet_oracle, resource_oracle, scenario
    )

function_images = images.all_ai_images 


# Configure scheduler with filtering predicates
predicates = []
predicates.extend(
    Scheduler.default_predicates
)  # Basic resource and selector predicates
predicates.extend(
    [
        # CanRunPred(
        #     fet_oracle, resource_oracle
        # ),  # Filter nodes where function can execute efficiently
        # NodeHasAcceleratorPred(),  # Filter for nodes with hardware accelerators
        # NodeHasFreeGpu(),  # Filter for nodes with available GPU capacity
        # NodeHasFreeTpu(),  # Filter for nodes with available TPU capacity
    ]
)

# Get priorities for scoring nodes that pass predicates
priorities = vanilla.get_priorities()  # Default node scoring priorities priorities.md

# Scheduler configuration parameters
sched_params = {
    "percentage_of_nodes_to_score": 100,  # Percentage of nodes to consider after filtering
    "priorities": priorities,  # Node scoring functions
    "predicates": predicates,  # Node filtering functions
}

# Set workload pattern - constant rate of requests
benchmark = ConstantBenchmark(
    "mixed", duration=500, rps=100
)  # rps requests/second for duration seconds

# Initialize network topology and storage
storage_index = StorageIndex()  # Tracks data locations in the network
topology = urban_sensing_topology(
    ether_nodes, storage_index
)  # Build edge-fog-cloud network

# Debug: Print topology summary
print(f"\n=== TOPOLOGY SUMMARY ===")
print(f"Total nodes in topology: {len(topology.nodes)}")

# Separate compute nodes from network infrastructure
compute_nodes = []
infrastructure_nodes = []

for node_name, node in topology.nodes.items():
    # Convert node_name to string if it's not already
    name_str = str(node_name)
    
    # Check if it's a compute node vs infrastructure
    if any(keyword in name_str.lower() for keyword in ['link', 'switch', 'shared', 'registry']):
        infrastructure_nodes.append((node_name, node))
    else:
        compute_nodes.append((node_name, node))

print(f"Compute nodes: {len(compute_nodes)}")
print(f"Infrastructure nodes: {len(infrastructure_nodes)}")

# Analyze compute nodes specifically
print(f"\n=== COMPUTE NODES ANALYSIS ===")
compute_node_types = {}
for node_name, node in compute_nodes[:20]:  # Show first 20
    # Try different ways to get the architecture
    arch = getattr(node, 'arch', None)
    if arch is None:
        # Check labels for architecture info
        labels = getattr(node, 'labels', {})
        arch = labels.get('arch', 'Unknown')
    
    accelerators = getattr(node, 'accelerators', [])
    capacity = getattr(node, 'capacity', {})
    
    name_str = str(node_name)
    print(f"{name_str:<15s} - Arch: {str(arch):<10s} - Accelerators: {accelerators}")
    if capacity:
        print(f"                    Capacity: {capacity}")
    
    # Count node types
    arch_str = str(arch)
    if arch_str in compute_node_types:
        compute_node_types[arch_str] += 1
    else:
        compute_node_types[arch_str] = 1

print(f"\n=== COMPUTE NODE TYPE DISTRIBUTION ===")
for arch, count in compute_node_types.items():
    print(f"{arch}: {count} nodes")

# Check if devices properties are preserved
print(f"\n=== ORIGINAL DEVICES vs TOPOLOGY NODES ===")
print(f"Generated devices: {len(devices)}")
print(f"Ether nodes: {len(ether_nodes)}")
print(f"Topology compute nodes: {len(compute_nodes)}")

# Sample original devices
print(f"\n=== SAMPLE ORIGINAL DEVICES ===")
for i, device in enumerate(devices[:5]):
    print(f"Device {i}: {device.arch.name} - {device.accelerator.name} - {device.location.name}")

# Sample ether nodes  
print(f"\n=== SAMPLE ETHER NODES ===")
for i, ether_node in enumerate(ether_nodes[:5]):
    print(f"Ether {i}: {ether_node.name} - Arch: {getattr(ether_node, 'arch', 'Unknown')} - Accelerators: {getattr(ether_node, 'accelerators', [])}")
# Initialize simulation environment
env = Environment()

# Configure environment components
env.simulator_factory = AIPythonHTTPSimulatorFactory(
    get_raith21_function_characterizations(resource_oracle, fet_oracle)
)  # Function execution simulator
env.metrics = Metrics(
    env, log=RuntimeLogger(SimulatedClock(env))
)  # Performance metrics collection
env.topology = topology  # Network topology
env.faas = DefaultFaasSystem(
    env, scale_by_requests=True
)  # FaaS system with auto-scaling
env.container_registry = ContainerRegistry()  # Container image registry
env.storage_index = storage_index  # Data location tracking
env.cluster = SimulationClusterContext(env)  # Cluster abstraction for scheduler
env.scheduler = Scheduler(env.cluster, **sched_params)  # Function placement scheduler

# Create and run the simulation
sim = Simulation(env.topology, benchmark, env=env)
result = sim.run()  # Execute simulation until benchmark completion

# Extract metrics into dataframes for analysis
dfs = {
    "invocations_df": sim.env.metrics.extract_dataframe(
        "invocations"
    ),  # Function call records
    "scale_df": sim.env.metrics.extract_dataframe("scale"),  # Auto-scaling events
    "schedule_df": sim.env.metrics.extract_dataframe(
        "schedule"
    ),  # Scheduling decisions
    "replica_deployment_df": sim.env.metrics.extract_dataframe(
        "replica_deployment"
    ),  # Function replica placements
    "function_deployments_df": sim.env.metrics.extract_dataframe(
        "function_deployments"
    ),  # Function deployment specs
    "function_deployment_df": sim.env.metrics.extract_dataframe(
        "function_deployment"
    ),  # Function deployment events
    "function_deployment_lifecycle_df": sim.env.metrics.extract_dataframe(
        "function_deployment_lifecycle"
    ),  # Function lifecycle
    "functions_df": sim.env.metrics.extract_dataframe(
        "functions"
    ),  # Function definitions
    "flow_df": sim.env.metrics.extract_dataframe("flow"),  # Data flow between nodes
    "network_df": sim.env.metrics.extract_dataframe("network"),  # Network metrics
    "utilization_df": sim.env.metrics.extract_dataframe(
        "utilization"
    ),  # Resource utilization
    "fets_df": sim.env.metrics.extract_dataframe(
        "fets"
    ), } # Function execution time measurements
print(len(dfs))
# Print column names and info for each DataFrame
for df_name, df in dfs.items():
    print(f"\n{df_name} columns:")
    if df is not None and not df.empty:
        print(f"  Shape: {df.shape}")
        print("  Columns:")
        for col in df.columns:
            # Get a sample value to show data type
            sample = df[col].iloc[0] if len(df) > 0 else None
            sample_type = type(sample).__name__ if sample is not None else "N/A"
            print(f"    - {col}: {sample_type}")
    else:
        print("  Empty or None DataFrame")

# Configuration identifiers
device_id = f"d{num_devices}"  # d100 for 100 devices
rps_id = f"r{benchmark.rps}"   # r50 for 50 rps
settings_id = "mhfd_deployement_autoscale_default"  # Match the settings used in generate_devices()

# Construct directory names with configuration identifiers
data_dir = f"./data/{settings_id}_{device_id}_{rps_id}"
vis_dir = f"./Vis/{settings_id}_{device_id}_{rps_id}"

# Create output directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Save each DataFrame to a CSV file
for df_name, df in dfs.items():
    if df is not None and not df.empty:
        output_path = os.path.join(data_dir, f"{df_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {df_name} to {output_path}")
    else:
        print(f"Skipped {df_name} (empty or None DataFrame)")

print(f"All DataFrames saved to {data_dir}/ directory")

# Store original sys.argv
original_argv = sys.argv.copy()

# Set up arguments for the report generator
sys.argv = [
    sys.argv[0],  # Keep the original script name
    "--data-dir", data_dir,
    "--output-dir", vis_dir
]

# Run the report generator
print(f"Generating analysis reports from {data_dir}/ to {vis_dir}/")
report_main()

# Restore original sys.argv
sys.argv = original_argv
