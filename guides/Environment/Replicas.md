# FunctionReplica: Role, Creation, and Lifecycle in the FaaS Simulation

`FunctionReplica` is not a function but a class that represents a running instance of a serverless function on a specific node in the simulation. It is the fundamental unit of execution in the FaaS system.

## Definition and Structure

```python
class FunctionReplica:
    """
    A function replica is an instance of a function running on a specific node.
    """
    function: FunctionDeployment       # The function deployment this replica belongs to
    container: FunctionContainer       # The specific container configuration being executed
    node: NodeState                    # The node where this replica is running
    pod: Pod                           # Kubernetes-style pod representation for scheduling
    state: FunctionState = FunctionState.CONCEIVED  # Current lifecycle state
    simulator: 'FunctionSimulator' = None  # The simulator that handles execution behavior
```

Key properties:

```python
@property
def fn_name(self):
    return self.function.name  # Function name for identification

@property
def image(self):
    return self.container.image  # Container image being executed
```

## Creation and Initialization

`FunctionReplica` objects are created during the function deployment and scaling processes:

```python
# In DefaultFaasSystem.create_replica
def create_replica(self, fd: FunctionDeployment, fn: FunctionContainer) -> FunctionReplica:
    # Create a pod for scheduling
    pod = self.create_pod(fd, fn)

    # Get a simulator for this function
    simulator = self.env.simulator_factory.create(self.env, fn)

    # Create the replica (without a node yet)
    replica = FunctionReplica()
    replica.function = fd
    replica.container = fn
    replica.pod = pod
    replica.simulator = simulator
    replica.state = FunctionState.CONCEIVED

    return replica
```

## Lifecycle Stages

A `FunctionReplica` goes through several state transitions during its lifecycle:

1. **CONCEIVED**: Initial state when created, not yet assigned to a node
2. **STARTING**: Assigned to a node, containers being deployed and initialized
3. **RUNNING**: Fully operational and ready to handle requests
4. **SUSPENDED**: Temporarily inactive but not removed

## Node Assignment and Execution

After creation, the replica is assigned to a node through scheduling:

```python
# In DefaultFaasSystem.deploy_replica
def deploy_replica(self, fd: FunctionDeployment, fn: FunctionContainer, services: List[FunctionContainer]):
    # Create a new replica
    replica = self.create_replica(fd, fn)

    # Add to scheduling queue
    self.env.scheduler_queue.put(replica.pod)

    # Wait for scheduler to assign a node
    result = yield from self.schedule_replica(replica)

    # Assign node to replica
    node = self.env.topology.get_node(result.node_name)
    node_state = self.env.get_node_state(node)
    replica.node = node_state

    # Start the replica
    self.env.process(self.simulate_function_start(replica))
```

## Execution Process

Once assigned to a node, a replica goes through its execution lifecycle:

```python
# In DefaultFaasSystem.simulate_function_start
def simulate_function_start(self, replica: FunctionReplica):
    replica.state = FunctionState.STARTING

    # Deploy phase (container image download)
    yield from replica.simulator.deploy(self.env, replica)

    # Startup phase (container initialization)
    yield from replica.simulator.startup(self.env, replica)

    # Setup phase (function initialization)
    yield from replica.simulator.setup(self.env, replica)

    # Mark as running and ready to handle requests
    replica.state = FunctionState.RUNNING
```

## Handling Requests

When requests are routed to a running replica, they're handled through the simulator:

```python
# In DefaultFaasSystem.invoke
def invoke(self, request: FunctionRequest):
    # Find running replicas
    replicas = self.get_replicas(request.name, FunctionState.RUNNING)

    # Select a replica through load balancing
    replica = self.next_replica(request)

    # Execute the request on the selected replica
    yield from self.simulate_function_invocation(replica, request)

# In DefaultFaasSystem.simulate_function_invocation
def simulate_function_invocation(self, replica: FunctionReplica, request: FunctionRequest):
    # Use the replica's simulator to execute the request
    yield from replica.simulator.invoke(self.env, replica, request)
```

## Termination and Cleanup

When scaling down or removing functions, replicas are terminated:

```python
# In DefaultFaasSystem.scale_down
def scale_down(self, function_name: str, remove: int):
    # Find replicas to remove
    replicas = self.get_replicas(function_name, FunctionState.RUNNING)

    # Remove selected replicas
    for replica in replicas[:remove]:
        # Execute teardown logic
        yield from replica.simulator.teardown(self.env, replica)

        # Update state tracking
        self.remove_replica(replica)
```

## Resource Management Integration

During execution, the replica interacts with the resource management system:

```python
# In AIPythonHTTPSimulator.invoke
def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
    # Claim resources for this replica
    yield from self.claim_resources(env, replica, request)

    # Execute function
    yield env.timeout(fet)

    # Release resources
    yield from self.release_resources(env, replica, request)
```

## Summary

`FunctionReplica` is the central execution unit in the FaaS simulation, representing a specific instance of a function running on a node. It:

1. Is **created** during function deployment or scaling
2. Gets **assigned** to a node through the scheduling process
3. Goes through **lifecycle phases** (deployment, startup, setup)
4. **Handles requests** through its associated simulator
5. **Claims and releases resources** during execution
6. Gets **terminated** during scaling down or function removal

This class acts as the bridge between the high-level FaaS system operations and the low-level execution behavior simulated by `FunctionSimulator` implementations, making it a critical component in the simulation framework.

# Function Replication and Pod Usage in the FaaS Simulation

## Why Functions Get Replicated

Functions in the FaaS system are replicated for three primary reasons:

1. **Handling Increased Load**: When request volume exceeds the capacity of existing replicas

   ```python
   # In FaasRequestScaler.run
   if invocations > self.threshold * len(running_replicas):
       # Scale up due to high load
       yield from faas.scale_up(self.fn.name, scale_replicas)
   ```

2. **Meeting Target Performance**: To maintain specific metrics like average requests per replica

   ```python
   # In AverageFaasRequestScaler.run
   average = invocations / running
   if average > self.threshold:
       # Scale up to reduce load per replica
       desired_replicas = int(math.ceil(average / self.threshold * running))
   ```

3. **Initial Deployment**: To meet minimum replica requirements defined in scaling configuration
   ```python
   # In DefaultFaasSystem.deploy
   scale_min = fd.scaling_config.scale_min
   for _ in range(scale_min):
       yield from self.deploy_replica(fd, fn, services)
   ```

## When Functions Get Replicated

Replication occurs under specific conditions:

1. **During Initial Deployment**: When a function is first deployed with `scale_min > 0`

2. **During Auto-Scaling Events**: Triggered by periodic checks of the scalers

   ```python
   # In FaasRequestScaler.run
   while True:
       yield self.env.timeout(self.alert_window)  # Check every X seconds
       # Check metrics and scale if needed
   ```

3. **After Cold Start**: When a request arrives for a scaled-to-zero function

   ```python
   # In DefaultFaasSystem.invoke
   if not replicas:  # No running replicas
       yield from self.poll_available_replica(request.name)  # Starts a replica
   ```

4. **Explicit Scaling Commands**: Through direct API calls (less common in the simulation)
   ```python
   yield from faas.scale_up(function_name, additional_replicas)
   ```

## Pods: Purpose and Usage

In the FaaS simulation, "Pod" is a Kubernetes-inspired concept that serves several critical functions:

### 1. Scheduling Interface

Pods are the interface between the FaaS system and the scheduler:

```python
# In DefaultFaasSystem.create_pod
def create_pod(self, fd: FunctionDeployment, fn: FunctionContainer) -> Pod:
    containers = [Container(fn.image, resources=Resource.from_dict(fn.resources))]
    spec = PodSpec(containers=containers)
    pod = Pod(name=f"{fd.name}-{uuid4()}", spec=spec)
    pod.metadata = {
        'function': fd.name,
        'image': fn.image,
        **fn.labels  # Function-specific labels added here
    }
    return pod
```

This creates a scheduling representation that the scheduler can evaluate.

### 2. Resource Requirements Specification

Pods specify the resources needed by a function:

```python
# Resource specification inside Pod creation
resources = Resource.from_dict(fn.resources)  # CPU, memory, GPU, etc.
containers = [Container(fn.image, resources=resources)]
```

This tells the scheduler what resources a replica needs.

### 3. Function Placement Constraints

Pods carry labels and constraints that affect placement:

```python
# Labels added to Pod metadata
pod.metadata = {
    'function': fd.name,
    'image': fn.image,
    'device.edgerun.io/accelerator': 'GPU',  # Hardware requirements
    'data.edgerun.io/input': 'video-stream-1'  # Data locality hints
}
```

These are used by scheduler predicates and priorities.

### 4. Tracking Throughout Lifecycle

Pods provide an identity for replicas during scheduling:

```python
# In DefaultFaasSystem.schedule_replica
def schedule_replica(self, replica: FunctionReplica):
    # Pod is used for scheduling
    result = yield from self.env.scheduler.schedule_pod(replica.pod)

    # Result refers back to the Pod for identity
    if result.pod.name != replica.pod.name:
        raise ValueError("Pod mismatch")
```

### 5. Kubernetes-Compatible Representation

This design mimics Kubernetes, enabling realistic scheduling logic:

```python
# In Scheduler.schedule_pod
def schedule_pod(self, pod: Pod) -> SchedulingResult:
    # Filter nodes with predicates
    for pred in self.predicates:
        nodes = [n for n in nodes if pred.passes_predicate(self.context, pod, n)]

    # Score remaining nodes with priorities
    for priority, scorer in self.priorities:
        scores = scorer.map_node_score(self.context, pod, nodes)
```

## Relationship Between Replicas and Pods

- Every `FunctionReplica` has exactly one `Pod` representation
- The `Pod` is created first for scheduling
- After scheduling, the `FunctionReplica` is assigned to the selected node
- The `Pod` remains part of the replica throughout its lifecycle

This design enables the simulation to model both the system-level orchestration (using Pods) and the execution-level behavior (using FunctionReplicas) in a manner consistent with real-world platforms like Kubernetes-based FaaS implementations.
