# The FaaS System and Its Scaling Mechanisms

The Function-as-a-Service (FaaS) system in the simulation is implemented through the [`DefaultFaasSystem`]system.py ) class, which manages function deployment, execution, and scaling.

## FaaS System Components

The FaaS system, instantiated in [`main.py`]main.py ) with:

```python
env.faas = DefaultFaasSystem(env, scale_by_requests=True)
```

Manages:

- **Function Deployments**: Collections of function containers that can be deployed
- **Function Replicas**: Running instances of functions on specific nodes
- **Function Execution**: Handling requests to invoke functions
- **Auto-scaling**: Adjusting the number of replicas based on demand

## How Scaling Works

The simulation implements three scaling approaches, controlled by parameters when creating the `DefaultFaasSystem`:

### 1. Request-Based Scaling (`scale_by_requests=True`)

```python
# In DefaultFaasSystem.__init__
if self.scale_by_requests:
    self.env.process(self.faas_scalers[fd.name].run())
```

This approach:

- Tracks raw request counts over short intervals
- Scales up when request rate exceeds `rps_threshold`
- Uses `scale_factor` to determine how many replicas to add
- Implemented by `FaasRequestScaler`

### 2. Average Request-Based Scaling (`scale_by_average_requests=False`)

```python
# In DefaultFaasSystem.__init__
if self.scale_by_average_requests_per_replica:
    self.env.process(self.avg_faas_scalers[fd.name].run())
```

This approach:

- Calculates average requests per running replica
- Compares to `target_average_rps` (default 200)
- Scales to maintain this target average
- Implemented by `AverageFaasRequestScaler`

### 3. Queue-Based Scaling (`scale_by_queue_requests_per_replica=False`)

```python
# In DefaultFaasSystem.__init__
if self.scale_by_queue_requests_per_replica:
    self.env.process(self.queue_faas_scalers[fd.name].run())
```

This approach:

- Monitors queue length of waiting requests per replica
- Compares to `target_queue_length` (default 75)
- Scales to maintain target queue size
- Implemented by `AverageQueueFaasRequestScaler`

## Scaling Process Details

Each scaling approach follows these steps:

1. **Monitoring**: Periodically checks metrics at `alert_window` intervals
2. **Decision Making**: Compares current values to thresholds
3. **Scaling Logic**:
   ```python
   if desired_replicas < len(running_replicas):
       # scale down
       scale = len(running_replicas) - desired_replicas
       yield from faas.scale_down(self.fn.name, scale)
   else:
       # scale up
       scale = desired_replicas - len(running_replicas)
       yield from faas.scale_up(self.fn.name, scale)
   ```

## Scaling Constraints

Scaling is bound by configuration limits:

- `scale_min`: Minimum number of replicas (default 1)
- `scale_max`: Maximum number of replicas (default 20)
- `scale_zero`: Whether to scale to zero when idle (default False)

The Raith21 simulation uses request-based scaling with automatic deployment:

```python
# Deploy functions initially
for fd in deployments:
    env.faas.deploy(fd)  # This scales up to fd.scaling_config.scale_min

# Later, auto-scaling takes over based on workload
```

This scaling system allows the simulation to realistically model how serverless platforms like AWS Lambda or OpenFaaS adjust capacity in response to changing workloads while respecting resource constraints.

Similar code found with 1 license type
