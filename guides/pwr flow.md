### **STEP 1: DEVICE GENERATION PIPELINE**

```
Device Classes → Ether Nodes → Topology Nodes → Scheduler Context
     ↓               ↓              ↓               ↓
   specs         capacity      network          scheduling
```

```python
# main.py lines 72-74
devices = generate_devices(num_devices, cloudcpu_settings)  # Create device specs
ether_nodes = convert_to_ether_nodes(devices)              # Add capacity
topology = urban_sensing_topology(ether_nodes, storage_index)  # Network
```

Original Device objects have specs (`cores`, `ram`, `cpu`) but no `capacity` attribute. Ether conversion creates `Capacity(CPU: 4000 Memory: 8363810816)` objects needed for scheduling.

### **STEP 2: POWER MONITORING SYSTEM CREATION**

```
Power Profiles → Power Oracle → Power Metrics → Environment
      ↓              ↓             ↓              ↓
   device specs   calculations   storage      integration
```

```python
# main.py lines 82-85
power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
power_metrics = PowerMetrics()
env.power_oracle = power_oracle      # Attach to environment
env.power_metrics = power_metrics    # Attach to environment
```

Creates power calculation engine and data storage separate from main simulation metrics.

### **STEP 3: FUNCTION DEPLOYMENT SCALING**

```
Base Functions → Smart City Instances → Zone Deployments → Unique Names
      ↓                   ↓                    ↓              ↓
   4 functions        30 instances       zone mapping    no conflicts
```

**What we did:**

```python
# deployments.py lines 71-124
scenarios = {
    "default": {
        "resnet50-inference": 9,      # Create 9 instances
        "speech-inference": 8,        # Create 8 instances
        "resnet50-preprocessing": 8,  # Create 8 instances
        "resnet50-training": 5,       # Create 5 instances
    }
}
```

Single function instances wouldn't stress the system. Multiple instances create realistic workload distribution across our 100 nodes (reduced by predicates).

### **STEP 4: BACKGROUND PROCESS INTEGRATION**

```
SimPy Generator → Background Processes → FaaS System Start → Automatic Execution
       ↓                    ↓                   ↓                   ↓
   function def         list append         start() method      periodic calls
```

```python
# main.py lines 297-304
def power_monitoring_loop(env):
    while True:
        yield env.timeout(env.power_monitoring_interval)
        monitor_power_consumption(env)

env.background_processes.append(power_monitoring_loop)  # Add function, not process!
```

FaaS system expects generator functions, not Process objects. This lets the system manage the process lifecycle.

### **STEP 5: UTILIZATION DATA PIPELINE**

```
Running Replicas → Resource State → Get Utilization → Calculate Power → Log Metrics
       ↓               ↓              ↓               ↓                   ↓
   actual load     tracking       realistic values   physics EQ        CSV export
```

```python
# power.py lines 150-190
def monitor_power_consumption(env):
    for node in env.topology.get_nodes():
        if hasattr(node, 'capacity'):  # Only compute nodes
            utilization = get_current_utilization(env, node_name)  # Real data
            power_watts = env.power_oracle.predict_power(...)      # Physics
            env.metrics.log('power', {...})                       # Export
```

Gets actual utilization from running function replicas, based on device power profile & utilization.

### **STEP 6: DATA EXTRACTION AND ANALYSIS**

```
Simulation Events → Metrics Logger → DataFrames → CSV Files → Visualizations
       ↓                ↓              ↓          ↓            ↓
   runtime data      structured      pandas     export     analysis
```

```python
# main.py lines 312-340
dfs = {
    "power_df": sim.env.metrics.extract_dataframe("power"),
    "energy_df": sim.env.metrics.extract_dataframe("energy"),
    # ... other metrics
}
# Save to CSV and generate reports
```

Converts simulation events into analyzable data format for power consumption analysis.

## 🎯 **COMPLETE INTEGRATION FLOW**

```
1. SETUP PHASE:
   Device Generation → Ether Conversion → Topology Creation → Environment Setup
          ↓                  ↓                 ↓               ↓
      100 devices      capacity objects   network nodes   power oracles

2. DEPLOYMENT PHASE:
   Function Selection → Instance Creation → Zone Mapping → Replica Scaling
          ↓                   ↓               ↓             ↓
      4 base types       30 instances    unique names   1-3 replicas

3. EXECUTION PHASE:
   Request Generation → Scheduling → Replica Execution → Resource Tracking
          ↓               ↓            ↓                ↓
      47,862 calls   node placement   actual work   utilization data

4. MONITORING PHASE:
   Background Process → Utilization Query → Power Calculation → Data Logging
          ↓                   ↓                  ↓               ↓
      every 5s              values        physics model    metrics store

5. ANALYSIS PHASE:
   Simulation End → DataFrame Export → CSV Generation → Report Creation
          ↓              ↓               ↓               ↓
      500s runtime   10,605 samples   persistent data   visualizations
```

## 🔧 **KEY INTEGRATION POINTS**

**Environment Extensions:**

```python
env.power_oracle = power_oracle          # Calculation engine
env.power_metrics = power_metrics        # Data storage
env.power_monitoring_interval = 5.0      # Sample rate
env.background_processes = [...]         # Process management
```

**Data Flow:**

```python
Node Capacity → Replica Placement → Resource Usage → Power Calculation → CSV Export
```

**Process Lifecycle:**

```python
FaaS Start → Background Processes → Periodic Monitoring → Data Collection → Simulation End
```

This integration creates realistic power monitoring by connecting device specifications to actual workload execution and physics-based power models.
