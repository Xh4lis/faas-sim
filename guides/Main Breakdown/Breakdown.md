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

[Generators](https://github.com/M4hf0d/faas-sim/tree/master/ext/raith21/generators)

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
  - Function Execution Time (FET) oracle predicts how long functions will take _(Oracle: Uses Statistical Dist found in [fet.py](https://github.com/M4hf0d/faas-sim/blob/master/ext/raith21/fet.py))_
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

- predicates: filtering functions that determine whether a node is eligible to run a specific function/workload. They act as binary "yes/no" filters that eliminate unsuitable nodes early in the scheduling process.
- Defines predicates (filters) for node selection
- Sets priorities for scoring suitable nodes
- Configures scheduler parameters

Predicates are executed every time the scheduler attempts to place a function on a node. This happens:

- When new function deployments are created
- When auto-scaling creates new function replicas
- When the scheduler needs to find a replacement node for a function

### Segment 6: Benchmark and Topology Creation (Lines 54-58)

```python
benchmark = ConstantBenchmark('mixed', duration=500, rps=700)
storage_index = StorageIndex()
topology = urban_sensing_topology(ether_nodes, storage_index)
```

- Creates a constant workload with 700 requests per second, the type of functions to be executed ["in this case mixed" ](https://github.com/M4hf0d/faas-sim/blob/master/ext/raith21/benchmark/constant.py)
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

- Set up the simulation [environment](https://github.com/M4hf0d/faas-sim/blob/master/guides/Environment/Environment.md) (core point and central link of the simulation of the simulation):

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

Extract and display simulation metrics

- Creates dictionary of dataframes for different metric types
- Prints summary of each dataframe (columns, shapes, etc.)
