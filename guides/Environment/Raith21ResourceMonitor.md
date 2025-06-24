# Raith21ResourceMonitor:

The `Raith21ResourceMonitor` provides a specialized approach to tracking resource utilization in the simulation, focusing on accurate CPU usage based on actual function execution timelines:

## Core Functionality

1. **Time-Window Based**: Monitors resource usage in 1-second windows (from `start_ts` to `end_ts`)

2. **Execution Trace Analysis**: Instead of using claimed resources, it examines actual function execution traces within each time window

3. **Overlap Calculation**: For each function call, it calculates exactly how much time the function was executing within the current window:

   ```
   last_start = max(window_start, call_start)
   first_end = min(window_end, call_end)
   overlap = first_end - last_start
   ```

4. **CPU Usage Aggregation**: Combines all function execution durations and scales by the function's CPU resource profile:

   ```
   sum = np.sum(trace_execution_durations)
   cpu_usage = sum * replica_usage['cpu']
   ```

5. **Resource Capping**: Enforces realistic CPU utilization by capping at 100%:
   ```
   window = ResourceWindow(replica, min(1, cpu_usage))
   ```

## Key Benefits

- **Higher Accuracy**: Measures actual execution time rather than claimed resources
- **Time-Aware**: Precisely accounts for partial executions within monitoring windows
- **Overlapping Functions**: Correctly handles multiple function calls within the same window
- **Realistic Modeling**: Prevents unrealistic CPU utilization values (>100%)

This monitor provides a more accurate foundation for energy modeling because it tracks exactly when and how long functions run, essential for accurate power × time calculations.
