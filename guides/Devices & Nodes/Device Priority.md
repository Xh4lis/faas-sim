# Understanding Function Placement Priorities in the Raith21 Simulation

The line `priorities = vanilla.get_priorities()` sets up the criteria for scoring and ranking nodes when placing functions. These priorities determine which nodes are preferred after filtering out unsuitable nodes through predicates.

## What Priorities Are Used

Looking at vanilla.py, the default priorities returned are:

```python
def get_priorities():
    return [
        (1, BalancedResourcePriority()),
        (1, ImageLocalityPriority()),
    ]
```

Each priority has two components:

1. A weight (the `1` values) that determines its relative importance
2. A priority class that implements the scoring logic

## How Each Priority Works

### 1. BalancedResourcePriority

This priority favors nodes that would maintain balanced resource utilization after placing a function.

**Calculation logic:**

- Looks at CPU and memory usage across all nodes
- Calculates how adding the new function would affect resource balance
- Prefers nodes where placing the function maintains an even CPU:memory usage ratio
- Helps prevent situations where one resource type becomes a bottleneck

### 2. ImageLocalityPriority

This priority favors nodes that already have the container image for the function cached locally.

**Calculation logic:**

- Checks if a node already has the container image from previous runs
- Gives higher scores to nodes with the image already present
- Reduces startup time by avoiding container image pulls
- Reduces network bandwidth consumption

## How Priorities Are Applied

1. **Individual Scoring**: Each priority runs its `map_node_score()` method on every candidate node:

   ```python
   def map_node_score(self, context: ClusterContext, pod: Pod, node: Node) -> int:
       # Calculate a score for this specific node
   ```

2. **Score Normalization**: After all nodes are scored, the `reduce_mapped_score()` method:

   ```python
   def reduce_mapped_score(self, context, pod, nodes, node_scores):
       return _scale_scores(node_scores, context.max_priority)
   ```

   Scales scores to a consistent range so different priorities can be fairly combined.

3. **Weighted Combination**: Scores from different priorities are multiplied by their weights and summed:

   ```python
   total_score = (weight1 * score1) + (weight2 * score2)
   ```

4. **Final Ranking**: Nodes are sorted by their total scores, and the highest-scoring node is selected for function placement.

## Alternative Priorities Available

The codebase also offers more advanced priorities in other scheduling configurations:

1. **CapabilityMatchingPriority**: Matches specific hardware capabilities to function requirements
2. **ExecutionTimePriority**: Favors nodes with faster execution times for the function
3. **ContentionPriority**: Evaluates resource contention and interference between co-located functions

Each of these could be added to create more sophisticated scheduling strategies beyond the default "vanilla" approach.

## Integration Point for Energy-Aware Scheduling

To implement energy-aware scheduling, you could add a new `EnergyEfficiencyPriority` class that would:

- Calculate expected energy consumption for running a function on each node
- Score nodes based on energy efficiency
- Prioritize placements that minimize overall energy consumption or carbon footprint
