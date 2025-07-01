## 🔧 **DETAILED FILE-BY-FILE INTEGRATION BREAKDOWN**

Let me show you exactly what you added to each file to achieve this power monitoring integration:

---

## 📁 **core.py - Environment Extensions**

### **What we Added:**

```python
# Added power monitoring attributes to Environment class
def __init__(self, initial_time=0):
    # ... existing code ...
    self.power_oracle = None      # NEW: Power calculation engine
    self.power_metrics = None     # NEW: Power data storage
    self.power_monitoring_interval = 5.0  # NEW: Sample every 5 seconds
```

- **Central Integration Point**: Environment is the hub connecting all simulation components
- **Global Access**: Any component can now access `env.power_oracle` and `env.power_metrics`
- **Configuration**: `power_monitoring_interval` controls how often power is sampled

---

## 📁 **metrics.py - Power Data Storage**

### **What we Added:**

```python
# Completely new PowerMetrics class
class PowerMetrics:
    """Track power consumption and energy usage"""

    def __init__(self):
        self.power_samples = []           # Store all power measurements
        self.energy_accumulation = {}     # Track cumulative energy per node
        self.last_timestamp = {}          # Track timing for energy calculation

    def record_power_sample(self, timestamp, node_name, node_type,
                          cpu_util, gpu_util, network_util,
                          memory_util, power_watts):
        """Record a power measurement and calculate energy delta"""

        # Calculate energy since last measurement (Power × Time)
        if node_name in self.last_timestamp:
            time_delta = timestamp - self.last_timestamp[node_name]
            energy_joules = power_watts * time_delta
            self.energy_accumulation[node_name] += energy_joules

        # Store comprehensive sample
        sample = {
            'timestamp': timestamp,
            'node': node_name,
            'node_type': node_type,
            'cpu_utilization': cpu_util,
            'gpu_utilization': gpu_util,
            'network_utilization': network_util,
            'memory_utilization': memory_util,
            'power_watts': power_watts,
            'energy_delta_joules': energy_joules,
            'cumulative_energy_joules': self.energy_accumulation[node_name],
            'cumulative_energy_wh': self.energy_accumulation[node_name] / 3600.0
        }

        self.power_samples.append(sample)
        return sample
```

- **Energy Accumulation**: Calculates energy from power × time
- **Comprehensive Data**: Stores both power and utilization in one place
- **Efficient Storage**: Uses lists for fast appending during simulation
- **Multiple Units**: Provides both Joules and Watt-hours for analysis

---

## 📁 **power.py - Power Calculation Engine**

#### **A. Device Power Profiles**

```python
# Lines 45-120 - Physics-based power models for each device type
DEVICE_POWER_PROFILES = {
    "rpi4": {
        "idle": 2.7,      # Base power when idle
        "cpu_max": 3.8,   # Additional power at 100% CPU
        "gpu_max": 1.0,   # Additional power at 100% GPU
        "network_max": 1.0,
        "memory_max": 0.5
    },
    "xeongpu": {
        "idle": 75.0,     # Much higher base power
        "cpu_max": 205.0, # High-performance CPU
        "gpu_max": 275.0, # Dedicated GPU card
        "network_max": 10.0,
        "memory_max": 8.0
    },
    # ... 11 total device types
}
```

#### **B. Power Calculation Engine**

```python
# Lines 15-40 - Linear power model implementation
class Raith21PowerOracle:
    def predict_power(self, node_type, cpu_util, gpu_util=0.0,
                     network_util=0.0, memory_util=0.0):
        """Linear Model: P = P_idle + (U_cpu × P_cpu_max) + (U_gpu × P_gpu_max) + ..."""

        profile = self.power_profiles.get(node_type, default_profile)

        # Linear combination of resource utilizations
        power = profile["idle"]                          # Base power
        power += cpu_util * profile.get("cpu_max", 0.0) # CPU contribution
        power += gpu_util * profile.get("gpu_max", 0.0) # GPU contribution
        power += network_util * profile.get("network_max", 0.0)
        power += memory_util * profile.get("memory_max", 0.0)

        return max(power, profile["idle"])  # Never below idle
```

#### **C. Utilization Data Extraction**

```python
# Lines 140-180 - Get real utilization from running replicas
def get_current_utilization(env, node_name: str) -> Dict[str, float]:
    """Extract actual resource usage from simulation state"""

    total_cpu = 0.0
    total_memory = 0.0
    total_gpu = 0.0
    replica_count = 0

    # Iterate through all running function replicas
    if hasattr(env, 'faas') and env.faas:
        for deployment in env.faas.get_deployments():
            for replica in env.faas.get_replicas(deployment.name):
                if replica.node.name == node_name:  # Replica on this node
                    replica_util = env.resource_state.get_resource_utilization(replica)
                    if replica_util and not replica_util.is_empty():
                        total_cpu += replica_util.get_resource_utilization('cpu', 0.0)
                        total_memory += replica_util.get_resource_utilization('memory', 0.0)
                        total_gpu += replica_util.get_resource_utilization('gpu', 0.0)
                        replica_count += 1

    return {
        'cpu': min(1.0, total_cpu),           # Aggregate CPU usage
        'memory': min(1.0, total_memory),     # Aggregate memory usage
        'gpu': min(1.0, total_gpu),           # Aggregate GPU usage
        'network': min(1.0, total_cpu * 0.3) # Estimate network from CPU
    }
```

#### **D. Monitoring Loop**

```python
# Lines 190-230 - Periodic power monitoring
def monitor_power_consumption(env):
    """Sample power across all compute nodes"""

    for node in env.topology.get_nodes():
        if hasattr(node, 'capacity'):  # Only compute nodes, skip infrastructure
            node_name = node.name
            node_type = extract_node_type(node_name)  # "rpi4_3" → "rpi4"

            # Get real-time utilization
            utilization = get_current_utilization(env, node_name)

            # Calculate power using physics model
            power_watts = env.power_oracle.predict_power(
                node_type, utilization['cpu'], utilization['gpu'],
                utilization['network'], utilization['memory']
            )

            # Store in power metrics system
            sample = env.power_metrics.record_power_sample(
                env.now, node_name, node_type,
                utilization['cpu'], utilization['gpu'],
                utilization['network'], utilization['memory'],
                power_watts
            )

            # Log to main metrics for DataFrame extraction
            env.metrics.log('power', {
                'timestamp': env.now, 'node': node_name, 'node_type': node_type,
                'power_watts': power_watts, 'cpu_util': utilization['cpu'],
                'gpu_util': utilization['gpu'], 'network_util': utilization['network'],
                'memory_util': utilization['memory']
            })

            env.metrics.log('energy', {
                'timestamp': env.now, 'node': node_name,
                'energy_joules': sample['cumulative_energy_joules'],
                'energy_wh': sample['cumulative_energy_wh']
            })
```

<!-- ### **Why This Architecture:**

- **Physics-Based**: Uses real device specifications (#TODO To be determined), not arbitrary numbers
- **Real Data**: Extracts actual utilization from running function replicas
- **Comprehensive**: Tracks CPU, GPU, memory, and network separately
- **Efficient**: Single pass through topology, logarithmic time complexity -->

---

## 📁 ** deployments.py - Smart City Function Scaling**

#### **A. Function Instance Multiplier**

```python
# Lines 10-70 - Create multiple instances of each function
def create_smart_city_function_instances(
    base_deployments: Dict[str, FunctionDeployment],
    instance_counts: Dict[str, int]
) -> List[FunctionDeployment]:
    """Transform 4 base functions into 30 unique instances"""

    expanded_deployments = []

    for func_name, base_deployment in base_deployments.items():
        count = instance_counts.get(func_name, 1)

        for i in range(count):
            # Deep copy to avoid reference conflicts
            new_deployment = copy.deepcopy(base_deployment)

            # Force consistent scaling configuration
            new_deployment.scaling_config.scale_min = 1  # Always start with 1 replica
            new_deployment.scaling_config.scale_max = 3  # Can scale up to 3

            if i == 0:
                # Keep original name for first instance
                expanded_deployments.append(new_deployment)
            else:
                # Create zone-specific instances
                zone_names = ["downtown", "suburb", "industrial", "residential",
                             "commercial", "airport", "port", "university", ...]
                zone_name = zone_names[i-1] if i-1 < len(zone_names) else f"zone{i}"

                # CRITICAL: Create unique function name
                new_deployment.fn.name = f"{func_name}-{zone_name}"
                expanded_deployments.append(new_deployment)

    return expanded_deployments
```

#### **B. Scenario Configurations**

```python
# Lines 80-110 - Workload distribution scenarios
scenarios = {
    "default": {
        "resnet50-inference": 9,      # 9 inference instances across zones
        "speech-inference": 8,        # 8 speech processing instances
        "resnet50-preprocessing": 8,  # 8 preprocessing instances
        "resnet50-training": 5,       # 5 training instances
        # Total: 30 function instances instead of 4
    },
    "intensive": {
        "resnet50-inference": 15,     # More intensive workload
        "speech-inference": 12,
        "resnet50-preprocessing": 15,
        "resnet50-training": 8,
        # Total: 50 instances
    }
}
```

### **Why This Approach:**

- **Realistic Load**: 30 function instances stress the system like real smart city deployment
- **Geographic Distribution**: Zone names simulate real urban deployment patterns
- **Unique Names**: Prevents scheduling conflicts and enables per-zone monitoring
- **Scalable Design**: Each instance can scale 1-3 replicas based on demand

---

## 📁 **main.py - Integration**

#### **A. Power System Imports**

```python
# Lines 60-65 - New imports for power monitoring
from ext.mhfd.power import (
    Raith21PowerOracle,     # Power calculation engine
    DEVICE_POWER_PROFILES,  # Device power specifications
    monitor_power_consumption, # Monitoring function
    power_monitoring_loop   # SimPy process
)
from sim.metrics import PowerMetrics  # Power data storage
```

#### **B. Power System Initialization**

```python
# Lines 95-100 - Create power monitoring components
power_oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)  # Physics engine
power_metrics = PowerMetrics()                            # Data storage

# Lines 280-285 - Attach to environment
env.power_oracle = power_oracle      # Global access to power calculator
env.power_metrics = power_metrics    # Global access to power data
```

#### **C. Background Process Integration**

```python
# Lines 290-300 - SimPy process integration
def power_monitoring_loop(env):
    """Periodic power monitoring process"""
    while True:
        yield env.timeout(env.power_monitoring_interval)  # Wait 5 seconds
        monitor_power_consumption(env)                    # Sample all nodes

# CRITICAL: Add function, not process object
env.background_processes.append(power_monitoring_loop)  # FaaS system will manage
```

#### **D. Data Extraction Enhancement**

```python
# Lines 320-325 - Add power metrics to DataFrame extraction
dfs = {
    # ... existing dataframes ...
    "power_df": sim.env.metrics.extract_dataframe("power"),     # Power measurements
    "energy_df": sim.env.metrics.extract_dataframe("energy"),   # Energy accumulation
}
```

---

## 🎯 **CRITICAL INTEGRATION INSIGHTS**

### **1. Environment as Integration Hub**

```python
# Instead of scattered globals, everything goes through Environment
env.power_oracle = power_oracle      # ✅ Centralized access
env.power_metrics = power_metrics    # ✅ Centralized storage
env.power_monitoring_interval = 5.0  # ✅ Centralized configuration
```

### **2. Background Process Pattern**

```python

# RIGHT: Let FaaS system manage process lifecycle
env.background_processes.append(power_monitoring_loop)
```

### **3. Data Integration**

```python


# RIGHT: Extract from actual simulation state
utilization = get_current_utilization(env, node_name)    #  replica data
power = env.power_oracle.predict_power(node_type, ...)
```

### **4. Data Flow Architecture**

```python
# Real Utilization → Power Calculation → Dual Storage → DataFrame Export
replica_util = env.resource_state.get_resource_utilization(replica)  # Source
power_watts = env.power_oracle.predict_power(...)                    # Transform
env.power_metrics.record_power_sample(...)                          # Store 1
env.metrics.log('power', {...})                                     # Store 2
df = sim.env.metrics.extract_dataframe("power")                     # Extract
```
