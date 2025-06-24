#!/usr/bin/env python
# coding: utf-8
import logging
import random
import time
from tqdm import tqdm
import numpy as np
from skippy.core.scheduler import Scheduler  # Kubernetes-inspired scheduler
from skippy.core.storage import StorageIndex  # Tracks data locations in the network

# Import Raith21-specific modules
from ext.raith21 import images  # Container image definitions for AI workloads
from ext.raith21.benchmark.constant import ConstantBenchmark  # Defines constant rate workloads
from ext.raith21.characterization import get_raith21_function_characterizations  # Function behavior models
from ext.raith21.deployments import create_all_deployments  # Creates function deployment specs
from ext.raith21.etherdevices import convert_to_ether_nodes  # Converts device specs to network nodes
from ext.raith21.fet import ai_execution_time_distributions  # Function execution time data
from ext.raith21.functionsim import AIPythonHTTPSimulatorFactory  # Function execution simulator
from ext.raith21.generator import generate_devices  # Creates heterogeneous device specs
from ext.raith21.generators.cloudcpu import cloudcpu_settings  # Device configuration settings
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle  # Performance prediction oracles
from ext.raith21.predicates import CanRunPred, NodeHasAcceleratorPred, NodeHasFreeGpu, NodeHasFreeTpu  # Node filters
from ext.raith21.resources import ai_resources_per_node_image  # Resource usage data
from ext.raith21.topology import urban_sensing_topology  # Edge-fog-cloud network topology
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

# Create oracles for predicting execution times and resource requirements
fet_oracle = Raith21FetOracle(ai_execution_time_distributions)  # Function Execution Time oracle
resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)  # Resource usage oracle

# Set up function deployments and container images
deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())  # Function deployment specs
function_images = images.all_ai_images  # Available container images for functions

# Configure scheduler with filtering predicates
predicates = []
predicates.extend(Scheduler.default_predicates)  # Basic resource and selector predicates
predicates.extend([
    CanRunPred(fet_oracle, resource_oracle),  # Filter nodes where function can execute efficiently
    NodeHasAcceleratorPred(),  # Filter for nodes with hardware accelerators
    NodeHasFreeGpu(),  # Filter for nodes with available GPU capacity
    NodeHasFreeTpu()  # Filter for nodes with available TPU capacity
])

# Get priorities for scoring nodes that pass predicates
priorities = vanilla.get_priorities()  # Default node scoring priorities priorities.md

# Scheduler configuration parameters
sched_params = {
    'percentage_of_nodes_to_score': 100,  # Percentage of nodes to consider after filtering
    'priorities': priorities,  # Node scoring functions
    'predicates': predicates  # Node filtering functions
}

# Set workload pattern - constant rate of requests
benchmark = ConstantBenchmark('mixed', duration=500, rps=700)  # rps requests/second for duration seconds

# Initialize network topology and storage
storage_index = StorageIndex()  # Tracks data locations in the network
topology = urban_sensing_topology(ether_nodes, storage_index)  # Build edge-fog-cloud network

# Initialize simulation environment
env = Environment()

# Configure environment components
env.simulator_factory = AIPythonHTTPSimulatorFactory(
    get_raith21_function_characterizations(resource_oracle, fet_oracle))  # Function execution simulator
env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))  # Performance metrics collection
env.topology = topology  # Network topology
env.faas = DefaultFaasSystem(env, scale_by_requests=True)  # FaaS system with auto-scaling
env.container_registry = ContainerRegistry()  # Container image registry
env.storage_index = storage_index  # Data location tracking
env.cluster = SimulationClusterContext(env)  # Cluster abstraction for scheduler
env.scheduler = Scheduler(env.cluster, **sched_params)  # Function placement scheduler

# Create and run the simulation
sim = Simulation(env.topology, benchmark, env=env)
result = sim.run()  # Execute simulation until benchmark completion

# Extract metrics into dataframes for analysis
dfs = {
    "invocations_df": sim.env.metrics.extract_dataframe('invocations'),  # Function call records
    "scale_df": sim.env.metrics.extract_dataframe('scale'),  # Auto-scaling events
    "schedule_df": sim.env.metrics.extract_dataframe('schedule'),  # Scheduling decisions
    "replica_deployment_df": sim.env.metrics.extract_dataframe('replica_deployment'),  # Function replica placements
    "function_deployments_df": sim.env.metrics.extract_dataframe('function_deployments'),  # Function deployment specs
    "function_deployment_df": sim.env.metrics.extract_dataframe('function_deployment'),  # Function deployment events
    "function_deployment_lifecycle_df": sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),  # Function lifecycle
    "functions_df": sim.env.metrics.extract_dataframe('functions'),  # Function definitions
    "flow_df": sim.env.metrics.extract_dataframe('flow'),  # Data flow between nodes
    "network_df": sim.env.metrics.extract_dataframe('network'),  # Network metrics
    "utilization_df": sim.env.metrics.extract_dataframe('utilization'),  # Resource utilization
    'fets_df': sim.env.metrics.extract_dataframe('fets')  # Function execution time measurements
}
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


# Extract final metrics in standard format
from .extract import extract_metrics  # Standardized metrics extraction
dfs = extract_metrics(sim)  # Process metrics into consistent format for analysis