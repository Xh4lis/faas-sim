# Recommended Depth for Energy Modeling Implementation

Based on the codebase and your goal of implementing energy modeling, I recommend focusing on the following depth:

## Appropriate Level of Understanding

### 1. Core System Components (High Priority)

- ✅ **`ResourceState` and `ResourceMonitor` classes**: Thoroughly understand how resource utilization is tracked
- ✅ **`Environment` class**: Understand how to extend it with energy models (similar to how `power_models` was added in the commit)
- ✅ **`Metrics` system**: Learn how metrics are collected and reported

### 2. Function Execution Flow (Medium Priority)

- Understand the simulator's function lifecycle (especially resource claiming/releasing)
- Follow how `FunctionReplica` objects interact with resources

### 3. Device Characteristics (Medium Priority)

- Review how different node types are defined
- Understand the device generation process to add power profiles

### 4. Skip or Skim (Low Priority)

- Network topology details
- Scheduling predicates/priorities internals
- Benchmark implementation details

## Implementation Approach

1. **Create a simple EnergyModel class** that uses linear modeling with the metrics you have
2. **Extend Environment** to store your energy models (similar to the `power_models` dict)
3. **Hook into ResourceMonitor** to calculate energy consumption periodically
4. **Add energy metrics collection** to the metrics system

## Proposed Workflow

1. **First Phase**: Create a basic implementation that:

   - Defines device-specific energy coefficients
   - Calculates power based on resource utilization
   - Logs power values as metrics

2. **Second Phase**: Add energy accumulation over time:

   - Track energy (power × time) as simulation progresses
   - Implement per-function, per-node, and total energy metrics

3. **Third Phase (Optional)**: Enhance with ML models if needed:
   - Add support for importing trained models
   - Use more sophisticated modeling approaches

## Code Files to Focus On

1. core.py: To see how `power_models` was added to `Environment`
2. resource.py: To understand resource tracking
3. `ext/tmueller23/functionsim.py`: To see implementation of power prediction

This focused approach lets you implement energy modeling without getting lost in scheduling algorithm details or network topology complexities.
