# The Metrics Class: In-Depth Analysis

## What It Is

The `Metrics` class is a comprehensive instrumentation and trace logger for the FaaS simulation framework. It serves as the central system for:

1. **Logging simulation events** across different components
2. **Tracking resource utilization** of functions and nodes
3. **Recording performance metrics** like execution times, wait times, and scaling events
4. **Collecting data for post-simulation analysis**

## How It Works

### Core Components and Structure

1. **Initialization** (lines 23-30):

   ```python
   def __init__(self, env: Environment, log: RuntimeLogger = None) -> None:
       self.env: Environment = env
       self.logger: RuntimeLogger = log or NullLogger()
       self.total_invocations = 0
       self.invocations = defaultdict(int)
       self.last_invocation = defaultdict(int)
       self.utilization = defaultdict(lambda: defaultdict(float))
   ```

   - Takes the simulation environment and an optional logger
   - Initializes counters for invocations and utilization tracking
   - Uses a `RuntimeLogger` to actually store the records

2. **Generic Logging Method** (lines 32-33):

   ```python
   def log(self, metric, value, **tags):
       return self.logger.log(metric, value, **tags)
   ```

   - Provides the fundamental logging capability
   - `metric`: The measurement category (e.g., 'invocations', 'node_utilization')
   - `value`: The data to record
   - `**tags`: Metadata attributes for filtering/grouping

3. **Specialized Logging Methods** (lines 35-195):

   - Event-specific methods like:
     - `log_invocation`: Records function invocation details
     - `log_function_resource_utilization`: Tracks resource usage by function
     - `log_scaling`: Monitors autoscaling events
   - Each method transforms the specific event data into a standardized format

4. **Data Extraction** (lines 202-226):
   ```python
   def extract_dataframe(self, measurement: str):
       # Collects all records for a specific measurement
       # Converts to pandas DataFrame
       # Sets timestamp index
   ```
   - Transforms raw log records into analysis-ready pandas DataFrames
   - Groups by measurement type (e.g., 'invocations', 'function_utilization')

### Data Flow

1. **Simulation components** call specialized logging methods
2. These methods format data and call the generic `log()` method
3. The `log()` method passes data to the underlying `RuntimeLogger`
4. `RuntimeLogger` stores records with timestamps
5. At the end of simulation, `extract_dataframe()` creates pandas DataFrames
6. Analysis code uses these DataFrames for visualization and insights

### Key Measurement Types

| Measurement            | Purpose                     | Examples                        |
| ---------------------- | --------------------------- | ------------------------------- |
| `invocations`          | Function call details       | Execution time, wait time, node |
| `function_utilization` | Per-function resource usage | CPU, memory, GPU usage          |
| `node_utilization`     | Per-node resource usage     | Total CPU/memory utilization    |
| `scale`                | Autoscaling events          | Replica count changes           |
| `schedule`             | Scheduling decisions        | Node assignments                |
| `replica_deployment`   | Lifecycle events            | Deploy, setup, teardown         |
| `fets`                 | Function execution times    | Start/end timestamps            |
| `network`              | Network transfers           | Bytes transferred               |
| `flow`                 | Data flow between nodes     | Source, sink, bytes, duration   |

## Why It's Important

The `Metrics` class is critical to the simulation framework for several reasons:

1. **Observability**: It makes simulation behavior visible and measurable
2. **Analysis Foundation**: It provides the raw data needed for performance evaluation, bottleneck identification, and system optimization

3. **Standardization**: It ensures that all components record data in a consistent format that can be easily processed

4. **Time-Series Analysis**: By tracking events with timestamps, it enables temporal analysis of system behavior

5. **Component Integration**: It acts as a central point where different simulation components can report their activities without needing to know how data will be used

6. **Extensibility**: Its design makes it easy to add new metrics (like energy consumption) without changing existing code

## Extension Point for Energy Modeling

To add energy modeling, you would:

1. **Create new logging methods**:

   ```python
   def log_power_consumption(self, node_name, watts, **tags):
       self.log('power', {'watts': watts}, node=node_name, **tags)

   def log_energy_consumption(self, node_name, joules, cumulative_joules, **tags):
       self.log('energy', {
           'joules': joules,
           'cumulative': cumulative_joules
       }, node=node_name, **tags)
   ```

2. **Call these methods from your extended ResourceMonitor**:

   ```python
   # In ResourceMonitor.run()
   power = calculate_power(node, utilization)
   self.env.metrics.log_power_consumption(node.name, power)
   ```

3. **Extract energy data for analysis**:
   ```python
   power_df = metrics.extract_dataframe('power')
   energy_df = metrics.extract_dataframe('energy')
   ```

The `Metrics` class's flexibility and standardized approach to data collection makes it the perfect foundation for adding energy consumption tracking to the simulation framework.

# Source of CPU Utilization in the Resource Tracking System

The `__calculate_util` method in the `Metrics` class calculates CPU utilization as a percentage by dividing the actual CPU resource usage by the node's CPU capacity. Let me explain where this CPU utilization data comes from:

## Data Flow for CPU Utilization

1. **Original Source**: CPU utilization starts when a function simulator claims resources during function execution:

   ```python
   # In function simulators (e.g., AIPythonHTTPSimulator, PowerPredictionSimulator)
   env.resource_state.put_resource(replica, 'cpu', cpu_value)
   ```

   This is where a specific amount of CPU is claimed for a function replica. The `cpu_value` comes from:

   - Function characterization data in resources.py or resources.py
   - For example: `('nano', 'resi5/resnet-inference-cpu'): FunctionResourceCharacterization(cpu=214.39...)`

2. **Storage in ResourceState**: When `put_resource` is called, the CPU value is stored in:

   ```python
   # In ResourceState.put_resource
   node_resources = self.get_node_resource_utilization(node_name)
   node_resources.put_resource(function_replica, resource, value)

   # Which stores in NodeResourceUtilization.__resources dictionary
   # And ultimately in ResourceUtilization.__resources
   ```

3. **Collection by ResourceMonitor**: The ResourceMonitor periodically samples this data:

   ```python
   # In ResourceMonitor.run()
   utilization = self.env.resource_state.get_resource_utilization(replica)
   ```

4. **Passed to Metrics**: The ResourceMonitor passes this data to the Metrics class:

   ```python
   # In ResourceMonitor.run()
   self.env.metrics.log_function_resource_utilization(replica, utilization)
   ```

5. **Calculation in \_\_calculate_util**: Finally, the `__calculate_util` method:
   - Gets the raw CPU value from `utilization.get_resource('cpu')`
   - Divides by `capacity.cpu_millis` (the total CPU capacity of the node)
   - The result is the CPU utilization as a fraction (which can be expressed as percentage)

## Example Flow

1. A function simulator claims 100 CPU units for a function: `put_resource(replica, 'cpu', 100)`
2. This value is stored in the ResourceState hierarchy
3. ResourceMonitor samples this value during its periodic run
4. The value is passed to `log_function_resource_utilization`
5. Inside `__calculate_util`, if the node's capacity is 1000 CPU units:
   - `utilization.get_resource('cpu')` = 100
   - `capacity.cpu_millis` = 1000
   - `cpu_util` = 100/1000 = 0.1 (or 10%)

This becomes part of the metrics record which is later available for analysis, visualization, or energy modeling calculations.
