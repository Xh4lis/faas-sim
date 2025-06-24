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
