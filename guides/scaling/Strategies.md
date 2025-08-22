# **High-Level Description of Three Scaling Strategies**

## **1. LPLT (Low-Power Long-Time) Strategy**

### **Philosophy**: Energy optimization through low-power device selection

- **Goal**: Minimize total energy consumption by using power-efficient devices
- **Trade-off**: Accepts longer execution times for lower power consumption

### **Scaling Logic**:

```python
# Moderate thresholds for energy efficiency
self.scale_up_threshold = 8     # Scale up at 8 RPS
self.scale_down_threshold = 3   # Scale down below 3 RPS
self.response_time_threshold = 1200  # Accept 1200ms response time
```

### **Node Selection Algorithm**:

1. **Power Efficiency Ranking** (watts idle power):

   ```python
   'rpi3': 1,    'nano': 2,    'rockpi': 3,    'coral': 4,    'rpi4': 5,
   'tx2': 6,     'nx': 7,      'nuc': 8,       'xeoncpu': 9,  'xeongpu': 10
   ```

2. **Selection Priority**:

   - **First**: Try consolidation on existing nodes with same deployment
   - **Second**: Bin-pack on nodes already running ANY replicas
   - **Third**: Use new low-power nodes only if needed

3. **Resource Thresholds**:
   ```python
   self.max_utilization_per_node = 0.85  # Don't overload beyond 85%
   self.min_utilization_for_new_node = 0.60  # Only new nodes if existing >60%
   ```

---

## **2. K8s (Kubernetes-Style) Strategy**

### **Philosophy**: Workload-aware balanced scheduling with device class matching

- **Goal**: Industry-standard approach with intelligent hardware-workload matching
- **Trade-off**: Balances performance, energy, and resource utilization

### **Scaling Logic** (HPA-style):

```python
# Aggressive scale-up, conservative scale-down
self.scale_up_threshold = 20      # Scale up at 20 RPS
self.scale_down_threshold = 5     # Scale down below 5 RPS
self.response_time_threshold = 500  # Target 500ms

# Multi-condition scaling (OR for up, AND for down)
Scale UP if: (load > 20 RPS) OR (response > 500ms) OR (CPU > 70%)
Scale DOWN if: (load < 5 RPS) AND (response < 250ms) AND (CPU < 30%)
```

### **Node Selection Algorithm** (Kubernetes Scheduler):

1. **Filtering Phase**: Remove unsuitable nodes (resource fit, node health)
2. **Scoring Phase** with device class awareness:

   ```python
   total_score = (
       least_requested_score * 6.0 +      # Spread workload
       device_class_score * 4.0 +         # Workload-hardware matching
       balanced_resource_score * 2.0      # CPU/memory balance
   ) / 12.0
   ```

3. **Device Class Mapping**:
   ```python
   function_requirements = {
       'resnet50-inference': 'gpu-accelerated',  # → nx, tx2, nano, xeongpu
       'resnet50-training': 'gpu-accelerated',   # → nx, tx2, nano, xeongpu
       'speech-inference': 'edge-optimized',     # → coral, nano, rockpi
       'python-pi': 'low-power',                 # → nano, coral, rockpi, rpi4
       'fio': 'high-compute'                     # → nuc, xeoncpu, xeongpu
   }
   ```

---

## **3. HPST (High-Performance Short-Time) Strategy**

### **Philosophy**: Energy optimization through fast execution on high-power devices

- **Goal**: Minimize total energy by reducing execution time (high_power × short_time)
- **Trade-off**: Higher instantaneous power for shorter total energy consumption

### **Scaling Logic**:

```python
# Aggressive performance thresholds
self.scale_up_threshold = 5     # Scale up at 5 RPS (very responsive)
self.scale_down_threshold = 2   # Scale down below 2 RPS
self.response_time_threshold = 500  # Target 500ms
self.max_response_time = 800    # Never exceed 800ms
```

### **Node Selection Algorithm**:

1. **Performance Ranking** (higher = better performance):

   ```python
   'xeongpu': 10, 'xeoncpu': 9, 'nuc': 8, 'nx': 7, 'tx2': 6,
   'rockpi': 5,   'nano': 4,    'rpi4': 3, 'rpi3': 2, 'coral': 1
   ```

2. **Strong High-Power Device Bias**:

   ```python
   # Heavy penalties for low-power devices
   if node_type in ['rpi3', 'rpi4']:
       base_performance -= 15  # Heavy penalty
   elif node_type in ['rockpi', 'coral']:
       base_performance -= 10  # Strong penalty

   # Major boosts for high-power devices
   if node_type in ['xeongpu', 'xeoncpu']:
       base_performance += 10  # Major boost
   ```

3. **Graduated Resource Thresholds** (easier qualification for high-power devices):
   ```python
   # Server-grade: 80% CPU/memory threshold (generous)
   # Mid-range (nuc, nx): 70% threshold (standard)
   # Low-power (rpi3, rpi4): 20% threshold (very strict)
   ```

---

## **Key Differences Summary**

| Aspect                   | LPLT                        | K8s                  | HPST               |
| ------------------------ | --------------------------- | -------------------- | ------------------ |
| **Primary Goal**         | Energy efficiency           | Balance              | Speed/Performance  |
| **Scale-up Trigger**     | 8 RPS                       | 20 RPS               | 5 RPS              |
| **Response Time Target** | 800m                        | 500ms                | 500ms              |
| **Node Preference**      | Low-power devices           | Workload-appropriate | High-power devices |
| **Resource Strategy**    | Consolidation + bin-packing | Balanced spreading   | Performance-first  |
