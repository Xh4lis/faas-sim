# Understanding ResourceState in the FaaS Simulation

`ResourceState` is a core component in the FaaS simulation framework that manages and tracks resource utilization across all computing nodes and function replicas. It serves as a centralized registry of resource consumption throughout the simulation.

## Basic Structure

```python
class ResourceState:
    node_resource_utilizations: Dict[str, NodeResourceUtilization]
```

The `ResourceState` class maintains a dictionary mapping node names to `NodeResourceUtilization` objects, creating a hierarchical structure:

- `ResourceState` → tracks all nodes
  - `NodeResourceUtilization` → tracks all function replicas on a node
    - `ResourceUtilization` → tracks specific resources used by each replica

## Key Methods

### 1. Managing Resources

```python
def put_resource(self, function_replica: FunctionReplica, resource: str, value: float):
    node_name = function_replica.node.name
    node_resources = self.get_node_resource_utilization(node_name)
    node_resources.put_resource(function_replica, resource, value)
```

This method adds a resource usage (e.g., CPU, memory, GPU) to a function replica on a specific node. This is called when:

- A function starts executing
- A function claims additional resources during execution

```python
def remove_resource(self, replica: 'FunctionReplica', resource: str, value: float):
    node_name = replica.node.name
    self.get_node_resource_utilization(node_name).remove_resource(replica, resource, value)
```

This method removes resource usage when:

- A function releases resources
- A function completes execution

### 2. Querying Resource Usage

```python
def get_resource_utilization(self, replica: 'FunctionReplica') -> 'ResourceUtilization':
    node_name = replica.node.name
    return self.get_node_resource_utilization(node_name).get_resource_utilization(replica)
```

Gets the current resources used by a specific function replica.

```python
def list_resource_utilization(self, node_name: str) -> List[Tuple['FunctionReplica', 'ResourceUtilization']]:
    return self.get_node_resource_utilization(node_name).list_resource_utilization()
```

Lists all resource utilization for all function replicas on a given node.

### 3. Node Management

```python
def get_node_resource_utilization(self, node_name: str) -> Optional[NodeResourceUtilization]:
    node_resources = self.node_resource_utilizations.get(node_name)
    if node_resources is None:
        self.node_resource_utilizations[node_name] = NodeResourceUtilization()
        node_resources = self.node_resource_utilizations[node_name]
    return node_resources
```

Gets (or creates if not existing) the `NodeResourceUtilization` object for a given node name.

## How It's Used in the Simulation

1. **During Function Execution:**

   ```python
   # When a function starts using CPU
   env.resource_state.put_resource(replica, 'cpu', 0.5)  # Using 50% CPU

   # When the function also uses GPU
   env.resource_state.put_resource(replica, 'gpu', 0.8)  # Using 80% GPU

   # When function is done
   env.resource_state.remove_resource(replica, 'cpu', 0.5)
   env.resource_state.remove_resource(replica, 'gpu', 0.8)
   ```

2. **By ResourceMonitor:**
   The `ResourceMonitor` regularly polls the `ResourceState` to collect metrics:

   ```python
   # Inside ResourceMonitor.run()
   utilization = env.resource_state.get_resource_utilization(replica)
   ```

3. **For Scheduling Decisions:**
   When scheduling functions, the system can check current resource utilization to make placement decisions.

4. **For Metrics Collection:**
   Resource usage data is tracked over time and logged for analysis.

## Key Points for Energy Modeling

For implementing energy modeling:

1. You can add power/energy as new resource types:

   ```python
   env.resource_state.put_resource(replica, 'power', calculated_power_watts)
   ```

2. You could track power at the node level by getting the total utilization:

   ```python
   node_util = env.resource_state.get_node_resource_utilization(node_name).total_utilization
   cpu_usage = node_util.get_resource('cpu')
   gpu_usage = node_util.get_resource('gpu')
   # Calculate energy based on these
   ```

3. The hierarchical structure allows tracking power at both individual function level and aggregate node level.

This resource tracking system provides the foundation needed to build energy consumption models in the simulation.
