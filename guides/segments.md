# Breakdown of the Raith21 Simulation Framework

The main.py script implements a Function-as-a-Service (FaaS) simulation that models serverless function execution across heterogeneous edge/cloud environments. Let's divide it into functional segments and explain how they work together.

## 1. Functional Segments of the Script

### Segment 1: Imports and Setup (Lines 1-26)

```python
#!/usr/bin/env python
# coding: utf-8
import logging
import random
import time
from tqdm import tqdm
import numpy as np
from skippy.core.scheduler import Scheduler
from skippy.core.storage import StorageIndex
# ... more imports ...
np.random.seed(1435)
random.seed(1435)
logging.basicConfig(level=logging.INFO)
```

- **Purpose**: Import required libraries and set up basic configuration
- **Key Components**:
  - Imports modules for scheduling, simulation, topology management, and metrics
  - Sets random seeds for reproducible simulation results
  - Configures logging

### Segment 2: Device and Topology Generation (Lines 27-30)

```python
num_devices = 100 #Min 24
devices = generate_devices(num_devices, cloudcpu_settings)
ether_nodes = convert_to_ether_nodes(devices)
```

- **Purpose**: Create the simulated device ecosystem
- **Key Components**:
  - Generates heterogeneous devices with different capabilities
  - Converts abstract device descriptions to network nodes

### Segment 3: Oracle Creation (Lines 32-33)

```python
fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
```

- **Purpose**: Create oracles that provide performance predictions
- **Key Components**:
  - Function Execution Time (FET) oracle predicts how long functions will take _(Oracle: Uses Statistical Dist found in fet.py)_
  - Resource oracle determines CPU/memory/etc requirements for functions

### Segment 4: Deployment Configuration (Lines 35-36)

```python
deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())
function_images = images.all_ai_images
```

- **Purpose**: Define function deployments and container images
- **Key Components**:
  - Creates deployment specifications for all functions
  - References container images with different architectures

### Segment 5: Scheduler Configuration (Lines 38-52)

```python
predicates = []
predicates.extend(Scheduler.default_predicates)
predicates.extend([
    CanRunPred(fet_oracle, resource_oracle),
    NodeHasAcceleratorPred(),
    NodeHasFreeGpu(),
    NodeHasFreeTpu()
])

priorities = vanilla.get_priorities()

sched_params = {
    'percentage_of_nodes_to_score': 100,
    'priorities': priorities,
    'predicates': predicates
}
```

- **Purpose**: Configure the function scheduling system
- **Key Components**:
  - predicates: filtering functions that determine whether a node is eligible to run a specific function/workload. They act as binary "yes/no" filters that eliminate unsuitable nodes early in the scheduling process.
  - Defines predicates (filters) for node selection
  - Sets priorities for scoring suitable nodes
  - Configures scheduler parameters

### Segment 6: Benchmark and Topology Creation (Lines 54-58)

```python
benchmark = ConstantBenchmark('mixed', duration=500, rps=700)
storage_index = StorageIndex()
topology = urban_sensing_topology(ether_nodes, storage_index)
```

- **Purpose**: Define workload pattern and network topology
- **Key Components**:
  - Creates a constant workload with 700 requests per second
  - Sets up storage tracking
  - Builds an urban sensing topology with the generated nodes

### Segment 7: Environment Initialization (Lines 60-73)

```python
env = Environment()
env.simulator_factory = AIPythonHTTPSimulatorFactory(
    get_raith21_function_characterizations(resource_oracle, fet_oracle))
env.metrics = Metrics(env, log=RuntimeLogger(SimulatedClock(env)))
env.topology = topology
env.faas = DefaultFaasSystem(env, scale_by_requests=True)
env.container_registry = ContainerRegistry()
env.storage_index = storage_index
env.cluster = SimulationClusterContext(env)
env.scheduler = Scheduler(env.cluster, **sched_params)
```

- **Purpose**: Set up the simulation environment
- **Key Components**:
  - Creates environment object
  - Configures function simulators with execution characteristics
  - Sets up metrics collection
  - Initializes FaaS system with auto-scaling
  - Creates container registry, cluster context, and scheduler

### Segment 8: Simulation Execution (Lines 75-76)

```python
sim = Simulation(env.topology, benchmark, env=env)
result = sim.run()
```

- **Purpose**: Run the simulation
- **Key Components**:
  - Creates simulation object with topology, benchmark, and environment
  - Runs simulation until benchmark completion

### Segment 9: Results Processing (Lines 78-97)

```python
dfs = {
    "invocations_df": sim.env.metrics.extract_dataframe('invocations'),
    # ... more dataframes ...
}
print(len(dfs))
# Print column names and info for each DataFrame
for df_name, df in dfs.items():
    print(f"\n{df_name} columns:")
    # ... printing details ...
```

- **Purpose**: Extract and display simulation metrics
- **Key Components**:
  - Creates dictionary of dataframes for different metric types
  - Prints summary of each dataframe (columns, shapes, etc.)

### Segment 10: Final Metrics Extraction (Lines 99-100)

```python
from .extract import extract_metrics
dfs = extract_metrics(sim)
```

- **Purpose**: Extract final metrics in standardized format
- **Key Components**:
  - Imports extraction module
  - Processes metrics into structured dataframes

## 2. Flow and Lifecycle of the Simulation

### Phase 1: Initialization

1. **Import Libraries**: Import required modules for simulation
2. **Set Random Seeds**: Ensure reproducibility of results
3. **Generate Devices**: Create devices with different hardware capabilities
4. **Create Oracles**: Set up systems for predicting execution times and resource usage
5. **Define Deployments**: Create specifications for function deployments
6. **Configure Scheduler**: Set up scheduling logic with predicates and priorities
7. **Define Benchmark**: Specify workload pattern and parameters
8. **Create Topology**: Build network topology connecting all devices

### Phase 2: Environment Setup

1. **Create Environment**: Initialize simulation environment
2. **Configure Function Simulators**: Set up simulators that model function execution
3. **Initialize Metrics Collection**: Prepare system for gathering performance data
4. **Set Up FaaS System**: Configure serverless platform with auto-scaling
5. **Create Container Registry**: Prepare registry for managing container images
6. **Initialize Cluster Context**: Create abstraction of cluster for scheduler
7. **Set Up Scheduler**: Attach scheduler to environment

### Phase 3: Simulation Execution

1. **Create Simulation Object**: Combine topology, benchmark, and environment
2. **Run Simulation**: Execute simulation until benchmark completion
   - Docker images are pulled to nodes
   - Functions are deployed based on scheduling decisions
   - Function requests are generated according to benchmark pattern
   - Functions execute with resource constraints and timing from oracles
   - Metrics are collected throughout simulation

### Phase 4: Results Processing

1. **Extract Metrics**: Gather metrics from simulation into dataframes
2. **Display Summary**: Print overview of collected metrics
3. **Process Final Metrics**: Extract standardized metrics for further analysis

## 3. Key Components and Their Relationships

- **Benchmark** defines the workload pattern (request rate, duration)
- **Topology** represents the network of devices and connections
- **Environment** ties together all simulation components
- **FaasSystem** handles function deployment and invocation
- **Oracles** provide performance predictions for functions
- **Scheduler** makes placement decisions for function deployments
- **Metrics** collect and organize performance data

When you implement your own functions and topology:

1. Define your device types and capabilities
2. Create function execution characteristics
3. Set up appropriate scheduling logic
4. Define your network topology
5. Configure your workload pattern
6. Run simulation and analyze results

The simulation follows a trace-driven approach where real-world measurements inform execution times and resource usage, making it realistic while still allowing for experimentation with different configurations.

Similar code found with 1 license type
