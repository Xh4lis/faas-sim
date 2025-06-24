# Function Lifecycle in the FaaS Simulation Framework

The function lifecycle in the FaaS simulation framework models how serverless functions operate in real-world environments, with special attention to resource management. Here's a comprehensive breakdown of this lifecycle, focusing on resource claiming and releasing:

## 1. Function Lifecycle Phases

The `FunctionSimulator` class defines five key lifecycle phases that every function goes through:

```python
class FunctionSimulator(abc.ABC):
    def deploy(self, env: Environment, replica: FunctionReplica): ...
    def startup(self, env: Environment, replica: FunctionReplica): ...
    def setup(self, env: Environment, replica: FunctionReplica): ...
    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest): ...
    def teardown(self, env: Environment, replica: FunctionReplica): ...
```

### Phase 1: Deployment

- **Purpose**: Models container image pulling to the node
- **Resource Impact**: Typically uses network bandwidth, but doesn't claim function-specific resources yet
- **Flow**: `env.faas.deploy(deployment)` → `simulate_function_start()` → `simulator.deploy()`

### Phase 2: Startup

- **Purpose**: Models container creation and initialization
- **Resource Impact**: Basic system resources allocated, but function isn't ready yet
- **Flow**: `simulate_function_start()` → `simulator.startup()`

### Phase 3: Setup

- **Purpose**: Models runtime initialization (e.g., loading ML models, initializing libraries)
- **Resource Impact**: Additional specialized resources might be claimed (e.g., model download)
- **Flow**: `simulate_function_start()` → `simulator.setup()`

### Phase 4: Invocation

- **Purpose**: Models actual function execution with request processing
- **Resource Impact**: Full resource claiming and releasing occurs here
- **Flow**: `env.faas.invoke()` → `simulate_function_invocation()` → `simulator.invoke()`

### Phase 5: Teardown

- **Purpose**: Models function termination (scaling down or cleanup)
- **Resource Impact**: All remaining resources are released
- **Flow**: `faas.remove_replica()` → `simulator.teardown()`

## 2. Resource Claiming and Releasing

The most important resource management happens in specialized watchdog implementations:

### Resource Claiming

In implementations like `HTTPWatchdog` and `AIPythonHTTPSimulator`, resource claiming follows this pattern:

```python
def claim_resources(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
    # 1. Get function's resource requirements (from characterization)
    resource_characterization = self.characterization.get_resources(replica.node.name, replica.image)
    cpu = resource_characterization.cpu
    gpu = resource_characterization.gpu
    memory = resource_characterization.ram
    # etc.

    # 2. Register resources with ResourceState
    env.resource_state.put_resource(replica, 'cpu', cpu)
    env.resource_state.put_resource(replica, 'gpu', gpu)
    env.resource_state.put_resource(replica, 'memory', memory)
    # etc.

    yield env.timeout(0)  # SimPy requirement for generators
```

This happens:

- Before actual function execution in `invoke`
- After a client request arrives
- Based on resource characterization specific to the function and node

### Resource Releasing

Similarly, resources are released after function execution:

```python
def release_resources(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
    # 1. Get same resource amounts that were claimed
    cpu = resource_characterization['cpu']  # Often stored when claiming
    gpu = resource_characterization['gpu']
    # etc.

    # 2. Remove resources from ResourceState
    env.resource_state.remove_resource(replica, 'cpu', cpu)
    env.resource_state.remove_resource(replica, 'gpu', gpu)
    # etc.

    yield env.timeout(0)
```

This happens:

- After function execution completes
- Using the same resource amounts that were initially claimed

## 3. Advanced Resource Management (from `PowerPredictionSimulator`)

In the `PowerPredictionSimulator`, resource management is enhanced for energy modeling:

```python
def claim_resources(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
    # 1. Get resource characterization
    resource_characterization = self.characterization.resource_oracle.get_resources(...)

    # 2. Extract resource values
    cpu = resource_characterization.cpu
    gpu = resource_characterization.gpu
    # etc.

    # 3. Calculate power consumption from resources
    features = np.array([cpu, gpu, blkio, net, ram])
    power = env.power_models[replica.node.name].predict(features.reshape(1, -1))[0]

    # 4. Store resources for later release
    self.resources[request.request_id] = {'cpu': cpu, 'gpu': gpu, ..., 'power': power}

    # 5. Register all resources including power
    env.resource_state.put_resource(replica, 'cpu', cpu)
    env.resource_state.put_resource(replica, 'gpu', gpu)
    env.resource_state.put_resource(replica, 'power', power)
    # etc.

    yield env.timeout(0)
```

## 4. Resource Monitoring

Throughout function execution, the `ResourceMonitor` periodically samples resource usage:

```python
def run(self):
    while True:
        # Wait for sampling interval
        yield self.env.timeout(self.reconcile_interval)

        # For each running function replica
        for deployment in faas.get_deployments():
            for replica in faas.get_replicas(deployment.name, FunctionState.RUNNING):
                # Get current resource utilization
                utilization = self.env.resource_state.get_resource_utilization(replica)

                # Log metrics
                self.env.metrics.log_function_resource_utilization(replica, utilization)
```

This sampling:

- Happens at regular intervals during simulation
- Captures point-in-time resource usage of each running function
- Creates time-series data for later analysis

## 5. Integration Point for Energy Modeling

Your energy modeling implementation would fit into this lifecycle by:

1. Extending a function simulator (like `HTTPWatchdog`) with energy-aware resource claiming
2. Calculating power based on resource utilization
3. Tracking energy accumulation in the `ResourceMonitor`
4. Logging energy metrics alongside other resource metrics

This aligns perfectly with the existing resource tracking system while adding the energy dimension.
