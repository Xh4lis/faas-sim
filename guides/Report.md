# Understanding and Extending the FaaS-Sim Framework: A Comprehensive Report

## Table of Contents

1. Introduction to FaaS-Sim
2. The Raith21 Simulation Framework
3. Simulation Architecture and Components
   - Environment
   - Resource State
   - Resource Monitor
   - Metrics System
   - Function Lifecycle
   - Deployment System
   - FaaS System and Scaling
   - Device and Topology Generation
4. Energy Modeling Integration
   - Research Background
   - Energy Data Sources
   - Energy Model Design
5. Implementation Roadmap
   - Phase 1: Energy Model Foundation
   - Phase 2: Integration with Resource Monitor
   - Phase 3: Metrics Collection and Analysis
   - Phase 4: Energy-Aware Scheduling
6. Expected Outcomes
7. Conclusion

## Introduction to FaaS-Sim

The FaaS-Sim framework is a sophisticated discrete-event simulation platform designed to model serverless function execution across heterogeneous edge-cloud environments. Unlike other simulators, it combines network modeling, function execution simulation, and resource management into a unified framework that can accurately represent real-world serverless deployments.

Key features of the framework include:

- Trace-driven simulation based on real-world measurements
- Support for heterogeneous hardware with diverse capabilities
- Realistic network topology and data transfer modeling
- Advanced scheduling and auto-scaling mechanisms
- Detailed metrics collection and analysis

## The Raith21 Simulation Framework

The Raith21 implementation, named after one of the framework's authors, is an example configuration that simulates AI workloads in an urban sensing scenario. This configuration deploys various deep learning models across a multi-tier edge-fog-cloud architecture.

### Key Components of Raith21

1. **Device Ecosystem**: A mix of edge devices (Raspberry Pi, Jetson Nano, Coral), fog nodes (Jetson NX, TX2, NUC), and cloud servers (Xeon CPU, Xeon GPU)

2. **Function Workloads**: Deep learning models including ResNet inference, MobileNet training, image classification, and object detection

3. **Urban Sensing Topology**: A geographic layout mimicking smart city deployment with neighborhoods, districts, and central clouds

4. **Resource Characterization**: Empirical measurements of resource usage for functions on different hardware

### Execution Flow in Raith21

The simulation follows a structured flow:

1. **Device Generation**: Creates a heterogeneous set of compute devices
2. **Oracle Initialization**: Sets up execution time and resource prediction oracles
3. **Deployment Creation**: Defines function deployments and their properties
4. **Scheduler Configuration**: Sets up scheduling policies with predicates and priorities
5. **Environment Setup**: Initializes the simulation environment and components
6. **Workload Generation**: Creates a pattern of function invocation requests
7. **Execution**: Runs the simulation with functions being deployed, scaled, and invoked
8. **Metrics Collection**: Records detailed performance and resource metrics throughout

## Simulation Architecture and Components

### Environment

The `Environment` class serves as the central hub connecting all simulation components:

```python
class Environment(simpy.Environment):
    def __init__(self, initial_time=0):
        super().__init__(initial_time)
        self.faas = None                # Function-as-a-Service system
        self.simulator_factory = None   # Creates function simulators
        self.topology = None            # Network topology
        self.storage_index = None       # Data location tracking
        self.benchmark = None           # Workload pattern
        self.cluster = None             # Cluster abstraction for scheduler
        self.container_registry = None  # Container image registry
        self.metrics = None             # Metrics collection
        self.scheduler = None           # Function placement scheduler
        self.node_states = dict()       # Node state tracking
        self.metrics_server = None      # Time-series metrics storage
        self.resource_state = None      # Resource utilization tracking
        self.resource_monitor = None    # Resource monitoring process
        self.background_processes = []  # Background processes
        self.degradation_models = {}    # Performance degradation models
```

This class provides:

- A unified interface for component access
- State management across the simulation
- Node state tracking and creation
- SimPy integration for discrete event simulation

As the central coordination point, the `Environment` class is an ideal integration point for new capabilities like energy modeling.

### Resource State

The `ResourceState` class maintains a hierarchical structure that tracks resource utilization across all compute nodes and function replicas:

```
ResourceState
├── NodeResourceUtilization (for node_1)
│   ├── ResourceUtilization (for function_1)
│   ├── ResourceUtilization (for function_2)
│   └── ...
├── NodeResourceUtilization (for node_2)
└── ...
```

Key methods:

- `put_resource(replica, resource, value)`: Claims resources for a function
- `remove_resource(replica, resource, value)`: Releases resources
- `get_resource_utilization(replica)`: Gets current resource usage
- `list_resource_utilization(node_name)`: Lists all utilization on a node

This system forms the foundation for resource tracking and, by extension, energy modeling.

### Resource Monitor

The `ResourceMonitor` is a background process that periodically samples and records resource utilization:

```python
def run(self):
    """SimPy process that runs throughout simulation, collecting resource metrics"""
    while True:
        # Wait for next collection interval
        yield self.env.timeout(self.reconcile_interval)

        # Iterate through all running function replicas
        for deployment in self.env.faas.get_deployments():
            for replica in self.env.faas.get_replicas(deployment.name, FunctionState.RUNNING):
                # Get current resource usage for this replica
                utilization = self.env.resource_state.get_resource_utilization(replica)

                # Log metrics
                self.env.metrics.log_function_resource_utilization(replica, utilization)
```

This class is crucial for energy modeling because:

- It regularly samples resource usage at fixed intervals
- It has access to complete resource utilization data
- It interfaces with the metrics system
- The periodic sampling creates natural time windows for energy calculation

### Metrics System

The `Metrics` class serves as a comprehensive instrumentation and trace logger:

```python
def log(self, metric, value, **tags):
    return self.logger.log(metric, value, **tags)
```

Key measurement types include:

- `invocations`: Function call details
- `function_utilization`: Per-function resource usage
- `node_utilization`: Per-node resource usage
- `scale`: Autoscaling events
- `schedule`: Scheduling decisions
- `network`: Network transfers

To extract data for analysis:

```python
def extract_dataframe(self, measurement: str):
    # Transforms logs into pandas DataFrames
```

This system provides the foundation for adding energy metrics to the simulation.

### Function Lifecycle

Functions in the simulation go through five key lifecycle phases, each with specific resource impacts:

1. **Deployment Phase**:

   - Image pulling to the node
   - Network bandwidth usage but minimal function resources

2. **Startup Phase**:

   - Container creation
   - Basic system resource allocation

3. **Setup Phase**:

   - Runtime initialization
   - Model loading and specialized resource claiming

4. **Invocation Phase**:

   - Request processing
   - Full resource claiming and releasing
   - Actual function execution

5. **Teardown Phase**:
   - Function termination
   - Resource cleanup and release

Resource claiming follows this pattern:

```python
def claim_resources(self, env, replica, request):
    # Get resource requirements
    resource_characterization = self.characterization.get_resources(replica.node.name, replica.image)

    # Register resources
    env.resource_state.put_resource(replica, 'cpu', resource_characterization.cpu)
    # ... other resources
```

This lifecycle provides multiple integration points for energy modeling, particularly during resource claiming and the periodic monitoring of active functions.

### Deployment System

Function deployments in the simulation are created through the `create_all_deployments` function:

```python
deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())
```

Each `FunctionDeployment` contains:

- Function name and ID
- Container image to use
- Resource requirements
- Scaling configuration
- Scheduling constraints

The related container images are defined in `images.all_ai_images`:

```python
function_images = images.all_ai_images
```

These define:

- Image name (e.g., 'faas-workloads/resnet-inference-gpu')
- Image properties (size, supported architectures)
- Tags (e.g., 'latest')

This system enables hardware-aware placement by matching function requirements with node capabilities.

### FaaS System and Scaling

The `DefaultFaasSystem` class manages function deployment, execution, and scaling:

```python
env.faas = DefaultFaasSystem(env, scale_by_requests=True)
```

It implements three scaling approaches:

1. **Request-based Scaling**: Scales based on raw request count
2. **Average Request-based Scaling**: Maintains target requests per replica
3. **Queue-based Scaling**: Scales based on request queue length

When scaling up:

```python
def scale_up(self, fn_name: str, replicas: int):
    # Add more replicas while respecting scaling limits
```

When scaling down:

```python
def scale_down(self, function_name: str, remove: int):
    # Remove replicas while maintaining minimum count
```

This system determines how many function replicas are running at any given time, directly impacting overall energy consumption.

### Device and Topology Generation

The simulation creates devices with specific properties:

```python
@dataclass
class Device:
    id: str
    arch: Arch                # CPU architecture (ARM32, AARCH64, X86)
    accelerator: Accelerator  # Special hardware (NONE, GPU, TPU)
    cores: Bins               # CPU core count
    disk: Disk                # Storage type
    location: Location        # Deployment location
    connection: Connection    # Network connection
    cpu_mhz: Bins             # CPU frequency
    cpu: CpuModel             # CPU model
    ram: Bins                 # RAM capacity
```

The device generation process:

1. Creates abstract device descriptions
2. Converts them to Ether nodes for networking
3. Organizes them into a topology with geographic structure

This provides the hardware foundation on which energy models will operate.

## Energy Modeling Integration

### Research Background

Energy modeling in edge computing is becoming increasingly important as AI workloads proliferate across distributed systems. The DeepEn2023 energy dataset provides valuable real-world measurements specifically for edge AI workloads:

- Kernel-level energy measurements for basic operations
- Model-level energy consumption for state-of-the-art DNNs
- Application-level measurements for end-to-end scenarios

### Energy Data Sources

To implement accurate energy modeling, we can use data from:

1. **Device Manufacturer Documentation**:

   - NVIDIA Jetson power guides
   - Raspberry Pi power specifications

2. **Academic Papers and Datasets**:

   - DeepEn2023 energy dataset
   - Published benchmarks and measurements

3. **Typical Power Values for Edge Devices**:
   - Raspberry Pi 4: Base ~2.5W, CPU Max ~3.5W
   - Jetson Nano: Base ~2W, GPU Max ~2W
   - Jetson Xavier NX: Base ~5W, GPU Max ~15W
   - Xeon CPU: Base ~50W, Max ~125W
   - Xeon GPU: Base ~75W, GPU Max ~150W

### Energy Model Design

Our approach to energy modeling will use a linear resource-based model:

```python
class EnergyModel:
    def __init__(self, base_power=5.0, coefficients=None):
        # Default coefficients if none provided
        self.base_power = base_power  # Idle power in Watts
        self.coefficients = coefficients or {
            'cpu': 10.0,     # W per unit CPU utilization
            'memory': 0.01,  # W per MB
            'gpu': 15.0,     # W per unit GPU utilization
            'blkio': 0.05,   # W per MB/s
            'net': 0.08      # W per MB/s
        }

    def calculate_power(self, resource_utilization):
        """Calculate power consumption based on resource utilization"""
        power = self.base_power

        for resource, coef in self.coefficients.items():
            usage = resource_utilization.get_resource(resource) or 0
            power += usage * coef

        return power
```

This model:

- Uses device-specific base power values
- Applies resource-specific coefficients
- Can be calibrated with real-world measurements
- Calculates instantaneous power from resource utilization
- Can be extended to track energy over time (power × time)

## Implementation Roadmap

### Phase 1: Energy Model Foundation

**Goal**: Create the core energy model implementation

**Tasks**:

1. Create the `EnergyModel` class with device-specific profiles
2. Define device-specific power coefficients based on research data
3. Implement power calculation based on resource utilization
4. Add the models to the `Environment` class

**Key Components**:

```python
# Add to Environment.__init__
self.energy_models = {}  # Dict[node_name -> EnergyModel]

# Create energy models for device types
energy_models = {
    'nx': EnergyModel(base_power=5.0, coefficients={...}),
    'nano': EnergyModel(base_power=2.0, coefficients={...}),
    # ... other device types
}
```

### Phase 2: Integration with Resource Monitor

**Goal**: Track energy consumption over time

**Tasks**:

1. Extend `ResourceMonitor` to calculate power at each sampling interval
2. Track energy accumulation (power × time) between samples
3. Implement per-node and per-function energy tracking

**Key Components**:

```python
class EnergyAwareResourceMonitor(ResourceMonitor):
    def __init__(self, env, reconcile_interval=0.2, logging=True):
        super().__init__(env, reconcile_interval, logging)
        self.energy_consumption = defaultdict(float)  # Track total energy by node
        self.last_timestamp = 0

    def run(self):
        # Original monitoring logic
        # ...

        # Add energy calculation
        current_time = self.env.now
        elapsed = current_time - self.last_timestamp

        for node_name, node_util in self.env.resource_state.node_resource_utilizations.items():
            # Calculate power
            power = self.env.energy_models[node_name].calculate_power(node_util.total_utilization)

            # Track energy (power × time)
            energy_joules = power * elapsed
            self.energy_consumption[node_name] += energy_joules

            # Log metrics
            # ...

        self.last_timestamp = current_time
```

### Phase 3: Metrics Collection and Analysis

**Goal**: Record and analyze energy metrics

**Tasks**:

1. Add energy-specific metrics to the `Metrics` class
2. Implement energy DataFrame extraction
3. Create visualization and analysis tools for energy data

**Key Components**:

```python
# Add to Metrics class
def log_power_consumption(self, node_name, watts):
    self.log('power', {'watts': watts}, node=node_name)

def log_energy_consumption(self, node_name, joules, cumulative_joules):
    self.log('energy', {
        'joules': joules,
        'cumulative': cumulative_joules
    }, node=node_name)

# Analysis functions
def extract_energy_metrics(sim):
    energy_df = sim.env.metrics.extract_dataframe('energy')
    power_df = sim.env.metrics.extract_dataframe('power')

    total_energy = energy_df.groupby('node')['joules'].max().sum()
    invocations_df = sim.env.metrics.extract_dataframe('invocations')
    energy_per_invocation = total_energy / len(invocations_df) if len(invocations_df) > 0 else 0

    return {
        'energy_df': energy_df,
        'power_df': power_df,
        'total_energy': total_energy,
        'energy_per_invocation': energy_per_invocation
    }
```

### Phase 4: Energy-Aware Scheduling

**Goal**: Implement scheduling strategies that consider energy efficiency

**Tasks**:

1. Create an energy-aware scheduling priority
2. Implement predicates that filter nodes based on energy efficiency
3. Develop algorithms for energy-optimal function placement

**Key Components**:

```python
class EnergyEfficiencyPriority(Priority):
    def map_node_score(self, context: ClusterContext, pod: Pod, node: Node) -> float:
        # Predict energy consumption on this node
        function_image = pod.metadata['image']
        node_name = node.name

        # Get expected energy consumption
        expected_energy = predict_energy_consumption(function_image, node_name)

        # Give higher scores to energy-efficient nodes
        return 100 / (1 + expected_energy)  # Higher score = lower energy
```

## Expected Outcomes

By implementing this energy modeling approach, we expect to:

1. **Measure Energy Consumption**: Track power and energy usage across nodes and functions
2. **Compare Hardware Efficiency**: Quantify energy differences between device types
3. **Evaluate Scheduling Impact**: Assess how scheduling decisions affect energy consumption
4. **Optimize Function Placement**: Develop strategies that balance performance and energy
5. **Model Accuracy**: Validate the model against real-world measurements
6. **Scalable Analysis**: Extend to larger deployments and more complex scenarios

## Conclusion

The FaaS-Sim framework provides a robust foundation for simulating serverless function execution in heterogeneous edge-cloud environments. By extending it with energy modeling capabilities, we can gain valuable insights into the energy efficiency of different deployment strategies, hardware choices, and workload patterns.

Our phased implementation approach:

1. Builds a solid energy model foundation
2. Integrates with the existing resource monitoring system
3. Extends the metrics collection framework
4. Adds energy-aware scheduling capabilities
