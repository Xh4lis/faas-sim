#!/usr/bin/env python
# coding: utf-8
import logging
import random
import time
import os
from tqdm import tqdm
import numpy as np
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
print(devices.__str__())
