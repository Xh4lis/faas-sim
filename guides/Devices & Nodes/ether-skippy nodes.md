# Ether Nodes and Skippy Nodes: Understanding the Connection

In the FaaS simulation framework, Ether nodes and Skippy nodes represent the same physical devices, but from different perspectives of the system. Let me explain how they're related:

## Ether Nodes

**Ether nodes** come from the `ether` package and represent devices in a **network topology** perspective:

```python
# In ext/raith21/main.py
ether_nodes = convert_to_ether_nodes(devices)  # Convert to network topology nodes
topology = urban_sensing_topology(ether_nodes, storage_index)
```

Ether nodes primarily model:

- Network connectivity between devices
- Bandwidth and latency characteristics
- Physical/logical location of devices
- Data transfer capabilities

Ether nodes are used by the network simulation components to calculate data transfer times, bandwidth utilization, and network congestion.

## Skippy Nodes

**Skippy nodes** come from the `skippy` package and represent devices from a **resource scheduling** perspective:

```python
# In sim/skippy.py
def from_ether_node(node: EtherNode) -> Node:
    """Convert an ether node to a skippy node for scheduling."""
    capacity = Capacity(node.resources)
    n = Node(node.name, capacity=capacity)
    n.metadata = dict(node.properties)
    return n
```

Skippy nodes primarily model:

- Available resources (CPU, memory, GPU)
- Node properties for scheduling decisions
- Resource capacity and constraints
- Hardware capabilities for function placement

Skippy nodes are used by the scheduler to determine where to place function replicas based on resource requirements and constraints.

## How They're Paired Together

Yes, they're paired together, with a one-to-one correspondence between Ether nodes and Skippy nodes:

1. **Creation Flow**:

   ```
   Raw devices → Ether nodes → Skippy nodes
   ```

2. **Conversion Happens Here**:

   ```python
   # In SimulationClusterContext.__init__
   for node in env.topology.get_nodes():
       skippy_node = SkippyNode.from_ether_node(node)
       self.__nodes[node.name] = skippy_node
   ```

3. **Reference Preservation**:
   The `name` property is used as the key to match nodes between systems:

   ```python
   # Ether node with name "device-123"
   ether_node = topology.get_node("device-123")

   # Gets corresponding Skippy node
   skippy_node = env.cluster.get_node("device-123")
   ```

4. **Synchronization**:
   - When a function is scheduled to a Skippy node, it's actually deployed to the corresponding Ether node
   - Resource utilization tracked on the Ether node is reflected in scheduling decisions made with Skippy nodes

## Practical Example

When a function deployment request arrives:

1. The **scheduler** uses Skippy nodes to decide placement:

   ```python
   result = env.scheduler.schedule(pod, bound_nodes)
   target_node_name = result.node_name
   ```

2. The **FaaS system** uses the selected node name to find the Ether node:

   ```python
   node = env.topology.get_node(target_node_name)
   ```

3. The function is deployed to that Ether node:
   ```python
   replica = FunctionReplica(deployment.name, deployment.image, node)
   env.process(env.faas.simulate_function_start(replica))
   ```

## Key Integration Point for Energy Modeling

For energy modeling, you'll work with both representations:

- Resource tracking happens at the Ether node level (via `ResourceState`)
- Scheduling decisions using energy awareness would happen at the Skippy node level

This dual representation allows the simulation to separate concerns between network behavior and resource scheduling, while maintaining a unified view of the system.
