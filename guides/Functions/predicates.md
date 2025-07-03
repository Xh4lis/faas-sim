# Understanding Predicates in the Scheduler

Here's how the predicates work in this simulation:

1. **When Predicates Are Used**: Predicates are executed every time the scheduler attempts to place a function on a node. This happens:

   - When new function deployments are created
   - When auto-scaling creates new function replicas
   - When the scheduler needs to find a replacement node for a function

2. **How They're Defined**:

   ```python
   predicates.extend(Scheduler.default_predicates)
   predicates.extend([
       NodeHasAcceleratorPred(),
       NodeHasFreeGpu(),
   ])
   ```

3. **How They Function**: Each time scheduling occurs, the scheduler:
   - Takes all available nodes
   - Runs each predicate on each node to filter out unsuitable nodes
   - Only considers the remaining nodes for placement scoring

The `NodeHasAcceleratorPred()` and `NodeHasFreeGpu()` predicates specifically check if nodes have accelerators and available GPU capacity at the moment of scheduling, not just at initialization.

This is why your predicates can enforce dynamic constraints like "only place this function on nodes with free GPU capacity" throughout the entire simulation lifetime.
