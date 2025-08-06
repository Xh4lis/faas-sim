Following the completion of the monitoring infrastructure integration, I have successfully implemented a complete scaling framework with multiple strategies tailored for heterogeneous edge environments. The work focuses on energy-performance trade-offs in Function-as-a-Service (FaaS) deployments across diverse edge devices.

---

## **1. Edge Device Generator & Infrastructure Setup**

### **EdgeAI Device Generator (edgeai.py)**

I made a custom device generator for realistic more edge AI deployments. The key improvements include:

**Device Distribution Strategy:**

- **3 Intel NUC devices (2.5%)** - Edge coordinators
- **42 RPi devices (35%)** - IoT sensors and monitoring
- **18 Coral TPU devices (15%)** - AI inference acceleration
- **14 Nano GPU devices (12%)** - GPU-based inference
- **10 NX devices (8%)** - High-performance AI processing
- **30 RockPi devices (25%)** - General compute nodes
- **3 TX2 devices (2.5%)** - Specialized AI workloads

**Issue Resolved:** The original generator heavily favored high-performance Xeon devices (GPU/CPU) due to their lower index numbers in the generation process, causing unrealistic workload concentration on powerful cloud-like infrastructure. Our generator eliminates Xeon devices entirely and creates a edge computing environment with appropriate device heterogeneity.

**Rationale:** This distribution reflects real-world edge AI deployments where the majority of nodes are resource-constrained devices (RPi, embedded systems) with strategic placement of specialized AI accelerators (Coral TPU, Jetson devices) for compute-intensive tasks.

---

## **2. Smart City Deployment Scenarios**

### **Function Deployment Strategy (deployments.py)**

I implemented a smart city-focused deployment strategy with geographical distribution of function instances:

```python
"light": {
    "resnet50-inference": 4,      # Traffic cameras, security
    "speech-inference": 3,        # Voice kiosks, assistants
    "resnet50-preprocessing": 1,  # Image processing pipeline
    "resnet50-training": 2,       # Federated learning
    "python-pi": 3,              # IoT monitoring (CPU Based)
    "fio": 3,                    # Storage benchmarking
    # Total: 16 initial instances
}
```

**Design Decision:** Each function type is deployed with **initial replicas** rather than starting from zero. This addresses the "cold-start" problem in constant-rate workloads where scaling decisions might not respond quickly enough to prevent initial request queuing and elevated response times.

---

## **3. Scaling Architecture & Common Framework**

### **Base Autoscaler Pattern (`base_autoscaler.py`)**

I developed a common "framework" that all scaling strategies inherit from, implementing the **Template Method** design pattern:

**Core Functionality:**

- **Scaling Decision Engine** - Evaluates metrics and determines scale-up/down/no-action
- **Metrics Collection** - Records detailed scaling decisions and performance data
- **Resource Management** - Handles replica lifecycle and node allocation
- **Evaluation Framework** - Tracks energy efficiency and performance metrics

**Data Collection:** The framework automatically generates comprehensive datasets including:

- `scaling_decisions_df` - Timestamp, strategy, action, node selection, load metrics
- `autoscaler_detailed_metrics_df` - Response times, wait percentages, scaling triggers
- `power_df` & `energy_df` - Energy consumption and efficiency tracking

### **Metrics Collector (`ScalingMetricsCollector`)**

Implements real-time monitoring of:

- Response time percentiles (95th, 99th)
- Resource utilization per node type
- Energy consumption patterns
- Scaling events

---

## **4. Scaling Strategies Implementation**

### **4.1 High Performance Short Time (HPST) - "performance"**

**Philosophy:** Optimize total energy through execution speed - _high_power × short_time_

**Implementation Highlights:**

- **Node Selection:** Prioritizes highest-performance devices (NX > NUC > Nano > RockPi)
- **Device Ranking:** Performance-based hierarchy with GPU acceleration preference
- **Use Case:** Time-critical applications where speed reduces total energy consumption

### **4.2 Low Power Long Time (LPLT) - "power"**

**Philosophy:** Optimize total energy through device efficiency - _low_power × long_time_

**Implementation Highlights:**

- **Node Selection:** Prioritizes lowest idle power devices (Coral > Nano > RPi > RockPi)
- **Power Efficiency Focus:** Rankings based on watts/performance ratios
- **Use Case:** Battery-powered edge deployments where energy conservation is critical

### **4.3 Kubernetes-Style First Fit (K8s FF) - "kubernetes"**

**Philosophy:** Kubernetes Default Scheduler with device class awareness

**Implementation Highlights:**

- **Weighted Scoring:** LeastRequestedPriority (6.0) + DeviceClassMatching (4.0) + BalancedResource (2.0)
- **Device Class Mapping:** Function requirements matched to appropriate device classes
- **Kubernetes Compatibility:** Mimics Kubernetes 1.26+ Dynamic Resource Allocation
- **Use Case:** Production baseline with intelligent device-type awareness

### **4.4 Standard First Fit (Basic FF) - "basic"**

**Philosophy:** Simple resource-based allocation without optimization

**Implementation Highlights:**

- **Resource-Only Decisions:** CPU/memory availability-based selection
- **No Device Awareness:** Treats all nodes as homogeneous
- **Baseline Comparison:** Control group for strategy evaluation
- **Use Case:** Traditional container orchestration approach

---

<!-- ## **5. Evaluation Framework & Metrics**

### **Primary Evaluation Metrics:**

1. **Energy Efficiency Score** = `total_requests / total_energy_joules`
2. **Performance-Energy Trade-off** = `(1/response_time) / energy_per_second`
3. **QoS Satisfaction Rate** = `requests_meeting_sla / total_requests`

### **Secondary Metrics:**

- Energy per request (J/req), Peak power consumption (W)
- Response time percentiles, Throughput (req/s)
- Node utilization distribution, Scaling event frequency -->

### **Visualization Suite:**

The framework generates comprehensive analysis including:

- Scaling decision timelines with node selection rationale
- Response time evolution per function type
- Energy consumption patterns across device types
- Correlation analysis between scaling triggers and performance

---

## **6. Current Status & Next Steps**

### **Completed:**

- Custom edge device generator with realistic distributions
- Smart city deployment scenarios with geographic distribution
- Four distinct scaling strategies with energy-performance focus
- Comprehensive metrics collection and evaluation framework
- Automated analysis and visualization pipeline

### **In Progress:**

🔄 **Accurate Power Profiling** - Gathering precise power consumption data for device ranking
🔄 **Academic Validation** - Sourcing research papers for strategy validation and benchmarking

### **Immediate Next Steps:**

1. **Power Profile Validation:** Research and implement accurate power consumption models for:

   - Coral TPU devices (inference vs idle power)
   - Jetson family power characteristics under different workloads
   - RPi power consumption patterns for IoT workloads

2. **Academic :** Identify and adapt evaluation metrics, scaling strategies and parameters from research papers
<!--
   - Edge computing scaling papers (AWS Greengrass, Azure IoT Edge)
   - Energy-efficient computing research (NVIDIA Jetson benchmarks)
   - Container orchestration studies (Kubernetes HPA evaluation) -->

3. **Strategy Refinement:** Fine-tune thresholds and parameters based on realistic power profiles and benchmark comparisons

<!--
## **Technical Impact & Contributions**

This work provides a **comprehensive framework for evaluating energy-performance trade-offs** in heterogeneous edge computing environments. The implementation addresses key gaps in existing research:

- **Realistic Device Heterogeneity** - Moves beyond homogeneous cloud assumptions
- **Energy-Aware Scaling** - Considers power consumption in scaling decisions
- **Geographic Distribution** - Models real-world edge deployment patterns
- **Comprehensive Evaluation** - Multi-dimensional metrics for strategy comparison

The framework is designed to be **extensible** for future scaling strategy research and **reproducible** for comparative studies. -->

---

I would appreciate the opportunity to discuss these developments in more detail and gather your feedback on the next phase of research. Please let me know if you'd like me to prepare a more detailed technical presentation or if you have any questions about specific implementation aspects.

<!--
**Attachments:**

- Technical documentation and code samples available upon request
- Sample visualization outputs demonstrating framework capabilities -->
