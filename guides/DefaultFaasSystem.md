# DefaultFaasSystem in the FaaS Simulation

`DefaultFaasSystem` is the central component that implements the serverless function management system in the simulation. It's a comprehensive implementation that models how real-world FaaS platforms (like AWS Lambda, OpenFaaS, or Azure Functions) operate. Let me explain its core responsibilities and mechanisms in detail:

## Core Responsibilities of DefaultFaasSystem

### 1. Function Lifecycle Management

`DefaultFaasSystem` handles the complete lifecycle of functions:

```python
def deploy(self, fd: FunctionDeployment):
    # Register function, setup auto-scaling, deploy initial replicas

def invoke(self, request: FunctionRequest):
    # Find appropriate replica, execute function, record metrics

def remove(self, fn: FunctionDeployment):
    # Scale down all replicas, clean up resources

def suspend(self, function_name: str):
    # Temporarily suspend a function (similar to OpenFaaS idler)
```

These methods allow the simulation to realistically model how functions are deployed, executed, and removed in a real FaaS platform.

### 2. Replica Management and Scheduling

```python
def deploy_replica(self, fd: FunctionDeployment, fn: FunctionContainer, services: List[FunctionContainer]):
    # Create replica and submit to scheduler

def run_scheduler_worker(self):
    # Background process that handles replica placement on nodes

def create_replica(self, fd: FunctionDeployment, fn: FunctionContainer) -> FunctionReplica:
    # Instantiate function replica with appropriate simulator
```

These methods handle how function replicas are created, where they get placed, and how they're executed on specific nodes.

### 3. Auto-Scaling Implementation

`DefaultFaasSystem` supports three distinct auto-scaling approaches:

```python
if self.scale_by_requests:
    self.env.process(self.faas_scalers[fd.name].run())  # Scale by raw request count

if self.scale_by_average_requests_per_replica:
    self.env.process(self.avg_faas_scalers[fd.name].run())  # Scale by requests per replica

if self.scale_by_queue_requests_per_replica:
    self.env.process(self.queue_faas_scalers[fd.name].run())  # Scale by queue length
```

And implements the actual scaling operations:

```python
def scale_up(self, fn_name: str, replicas: int):
    # Add more replicas while respecting scaling limits

def scale_down(self, function_name: str, remove: int):
    # Remove replicas while maintaining minimum count
```

### 4. Request Handling and Load Balancing

```python
def invoke(self, request: FunctionRequest):
    # Get available replicas
    replicas = self.get_replicas(request.name, FunctionState.RUNNING)

    # Scale from zero if needed
    if not replicas:
        yield from self.poll_available_replica(request.name)

    # Load balance if multiple replicas available
    if len(replicas) > 1:
        replica = self.next_replica(request)
    else:
        replica = replicas[0]

    # Execute function and record metrics
    yield from simulate_function_invocation(self.env, replica, request)
```

This handles how incoming function requests are distributed across available replicas, including scaling from zero and load balancing.

## Internal State Management

`DefaultFaasSystem` maintains several important data structures:

```python
self.replicas = defaultdict(list)  # All replicas by function name
self.functions_deployments = dict() # All deployed functions
self.replica_count = dict()  # Count of active replicas per function
self.functions_definitions = Counter() # Tracks replica counts by container image
```

These allow it to track the state of all deployed functions and their replicas throughout the simulation.

## Critical Flow Examples

### Example 1: Handling an Invocation When No Replicas Exist

```python
# In DefaultFaasSystem.invoke:
replicas = self.get_replicas(request.name, FunctionState.RUNNING)
if not replicas:
    # Wait for a replica to become available
    yield from self.poll_available_replica(request.name)
```

This models the "cold-start" behavior of FaaS platforms, where an invocation must wait for an instance to be created if none exist.

### Example 2: Auto-Scaling Logic

```python
# In AverageFaasRequestScaler.run:
average = invocations / running  # Calculate average requests per replica
desired_replicas = int(math.ceil(average / self.threshold * running))
if desired_replicas < len(running_replicas):
    # Scale down if too many replicas
    yield from faas.scale_down(self.fn.name, len(running_replicas) - desired_replicas)
else:
    # Scale up if too few replicas
    yield from faas.scale_up(self.fn.name, desired_replicas - len(running_replicas))
```

This demonstrates how the system continuously monitors load and adjusts replica counts.

## Integration with Other Components

`DefaultFaasSystem` integrates with several other simulation components:

1. **Scheduler**: Determines node placement for replicas
2. **Metrics**: Records detailed performance metrics
3. **Simulator Factory**: Creates appropriate function simulators
4. **Network Topology**: Models data transfers between nodes

This makes it the central coordination point for serverless function behavior throughout the simulation.

# DefaultFaasSystem and FunctionSimulator: Relationship and Collaboration

The `DefaultFaasSystem` and `FunctionSimulator` classes represent different abstraction layers in the simulation framework that work together to create a complete serverless execution environment. Here's how they relate and work together:

## Hierarchical Relationship

1. **`DefaultFaasSystem`** (high-level): Manages the lifecycle of function deployments and handles system-wide concerns like scheduling, scaling, and routing.

2. **`FunctionSimulator`** (low-level): Simulates the detailed execution behavior of individual function instances, including resource usage and execution time.

## How They Work Together

The collaboration happens through a well-defined workflow:

1. `DefaultFaasSystem` **receives** a function invocation request:

   ```python
   # In DefaultFaasSystem.invoke
   def invoke(self, request: FunctionRequest):
       # Find appropriate function replica
       replicas = self.get_replicas(request.name, FunctionState.RUNNING)
       replica = self.next_replica(request)  # Load balance across replicas
   ```

2. `DefaultFaasSystem` **delegates** execution to the appropriate simulator:

   ```python
   # In DefaultFaasSystem.simulate_function_invocation
   def simulate_function_invocation(self, replica, request):
       simulator = replica.simulator  # This is a FunctionSimulator instance
       yield from simulator.invoke(self.env, replica, request)
   ```

3. `FunctionSimulator` **executes** the function with specific behavior:

   ```python
   # In AIPythonHTTPSimulator.invoke
   def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
       # Simulate execution details
       fet = self.characterization.sample_fet(replica.node.name)
       yield env.timeout(fet)  # Simulate execution time
       # Record metrics, etc.
   ```

4. `DefaultFaasSystem` **monitors** resource usage and makes scaling decisions:
   ```python
   # In FaasRequestScaler.run
   def run(self):
       while True:
           # Calculate desired replicas
           if desired_replicas > len(running_replicas):
               yield from faas.scale_up(self.fn.name, desired_replicas - len(running_replicas))
           else:
               yield from faas.scale_down(self.fn.name, len(running_replicas) - desired_replicas)
   ```

## Key Differences and Responsibilities

### DefaultFaasSystem

- **System-level management**: Orchestrates the entire FaaS platform
- **Function deployment**: Creates and places function replicas on nodes
- **Request routing**: Routes invocation requests to appropriate replicas
- **Auto-scaling**: Monitors metrics and scales functions based on demand
- **Scheduling**: Coordinates with the scheduler to place functions on nodes

### FunctionSimulator

- **Execution-level details**: Models how a specific function executes
- **Resource usage**: Claims and releases CPU, memory, etc., during execution
- **Execution time**: Models realistic function execution durations
- **Data transfers**: Simulates network traffic for data downloads/uploads
- **Custom behaviors**: Implements specific behaviors for different function types

## Implementation Approach

The system follows a **factory-based approach**:

1. `DefaultFaasSystem` uses a `SimulatorFactory` to create appropriate simulators:

   ```python
   # In DefaultFaasSystem.create_replica
   def create_replica(self, fd, container):
       simulator = self.env.simulator_factory.create(self.env, container)
       replica = FunctionReplica(fd.name, container.image, node, simulator=simulator)
   ```

2. The `SimulatorFactory` creates specialized simulators based on function type:

   ```python
   # In AIPythonHTTPSimulatorFactory.create
   def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
       workers = int(fn.labels['workers'])
       return AIPythonHTTPSimulator(queue, linear_queue_fet_increase, fn, self.fn_characterizations[fn.image])
   ```

3. This allows custom function behaviors while maintaining a consistent interface:
   ```python
   # All simulators share the same interface
   def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
       # Implementation varies by simulator type
   ```

## Summary

`DefaultFaasSystem` and `FunctionSimulator` work together in a hierarchical relationship where the FaaS system manages the high-level orchestration of functions and delegates the detailed execution behavior to specialized function simulators. This separation of concerns allows the simulation to model both system-level behaviors (scaling, scheduling) and function-specific behaviors (execution time, resource usage) in a modular and extensible way.
