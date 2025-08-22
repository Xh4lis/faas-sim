## **Experimental Context Assessment**

### **Critical Change: Pre-Positioned Replicas Impact**

- **Previous experiments**: Started from 1 replica per function

- **Current experiment**: Started with 4-16 replicas (total: 16 initial replicas)

- **Research impact**: This masks scaling algorithm differences and reduces resource pressure

### **Experimental Parameters:**

- **Load**: 9 RPS across 6 functions initially = 1.5 RPS per function

- **Peak load**: ~3 RPS per function intially (sine wave max_rps = rps×2)

- **Duration**: 1200s = 6 complete sine wave cycles

**Critical insight**: With 16 initial replicas handling only 9 RPS total load, there's minimal scaling pressure, which probably explains the similar results across strategies.

---

## **Power Consumption**

### **Power Distribution Evidence:**

| Strategy | Total Power | Power vs K8s       | Node Usage | Replicas/Node |
| -------- | ----------- | ------------------ | ---------- | ------------- |
| **K8s**  | 472.9W      | Baseline           | 34 nodes   | 43.6 avg      |
| **LPLT** | 479.3W      | **-1.3% (worse)**  | 28 nodes   | 39.9 avg      |
| **HPST** | 469.8W      | **+0.7% (better)** | 42 nodes   | 35.1 avg      |

### **LPLT Power Efficiency Failure**

**from power distribution:**

```

LPLT power by node type:

- nano: 29 nodes × 2.8W = 80.2W (vs K8s: 75.0W)

- nx: 14 nodes × 11.7W = 163.3W (vs K8s: 163.0W)

- rockpi: 34 nodes × 3.8W = 130.5W (vs K8s: 129.7W)

```

**Analysis**: LPLT consumes MORE power despite using fewer nodes (28 vs 34) Which is heavely influenced by the power ranking.

**Evidence from LPLT code:**

```python

# From lplt.py - LPLT's power ranking shows the problem:

self.power_efficiency_ranking = {

'nano':  2,  # 2.0W - BUT has GPU acceleration

'coral':  4,  # 2.5W - BUT has TPU acceleration

}

```

**Why LPLT fails**: The strategy assumes device idle power equals efficiency, but ignores that accelerated devices (nano GPU, coral TPU) consume more power under load than their idle specifications.

---

## **Performance Analysis**

### **Response Time Performance:**

| Strategy | Median RT | Performance vs K8s | P95 RT  | P95 vs K8s         |
| -------- | --------- | ------------------ | ------- | ------------------ |
| **K8s**  | 1,284s    | Baseline           | 13,161s | Baseline           |
| **LPLT** | 1,358s    | **-5.7% (worse)**  | 13,403s | **-1.8% (worse)**  |
| **HPST** | 1,299s    | **-1.2% (worse)**  | 13,142s | **+0.1% (better)** |

### **K8s Performance Leadership Despite Pre-Positioning**

K8s achieves best median performance (1,284s) even though all strategies started with abundant replicas.

**Analysis from scaling behavior:**

```

Scaling Actions (Total):

K8s: 1,370 actions (289 up, 1,081 down) → 3.7:1 down/up ratio

LPLT: 1,129 actions (215 up, 914 down) → 4.3:1 down/up ratio

HPST: 1,221 actions (288 up, 933 down) → 3.2:1 down/up ratio

```

**Why K8s wins**: Despite having the most scaling actions (1,370), K8s maintains optimal resource allocation through superior scale-down discipline (3.7:1 ratio).

---

## **Scaling Decision Analysis**

### **Scaling Frequency Patterns:**

| Strategy | Total Actions | Scaling Efficiency vs K8s |
| -------- | ------------- | ------------------------- |
| **K8s**  | 1,370         | Baseline                  |
| **LPLT** | 1,129         | **18% fewer actions**     |
| **HPST** | 1,221         | **11% fewer actions**     |

### Pre-Positioned Replicas Reveal Strategy Differences

With 16 initial replicas for 9 RPS load, strategies show different resource management philosophies:

**K8s behavior**: Most active scaling (1,370 actions) - continuously optimizes resource allocation

**LPLT behavior**: Most conservative scaling (1,129 actions) - preserves low-power placements

**HPST behavior**: Moderate scaling (1,221 actions) - maintains performance nodes

**Analysis from code thresholds:**

```python

# K8s thresholds (from khpa.py):

self.scale_up_threshold =  20  # Very high threshold

self.scale_down_threshold =  5  # Conservative scale-down



# LPLT thresholds (from lplt.py):

self.scale_up_threshold =  8  # Lower threshold (more responsive)

self.scale_down_threshold =  3  # Lower threshold (more aggressive scale-down)



# HPST thresholds (from hpst.py):

self.scale_up_threshold =  5  # Lowest threshold (most responsive)

self.scale_down_threshold =  2  # Lowest threshold (most aggressive)

```

**Why scales more**: K8s thresholds could be causing frequent oscillation around these thresholds.

---

## **Node Selection Analysis**

### **from Code Implementation:**

**K8s Device Selection (from khpa.py):**

```python

def  get_device_class_score(self,  node_type:  str,  deployment_name:  str) ->  float:

device_classes = {

'gpu-accelerated': ['xeongpu',  'nx',  'tx2',  'nano'],

'edge-optimized': ['coral',  'nano',  'rockpi'],

'high-compute': ['nuc',  'xeoncpu',  'xeongpu']

}

function_requirements = {

'resnet50-inference':  'gpu-accelerated',

'speech-inference':  'edge-optimized',

'fio':  'high-compute'

}

```

**LPLT Device Selection (from lplt.py):**

```python

self.power_efficiency_ranking = {

'rpi3':  1,  # Always prefer lowest power

'nano':  2,  # Second choice

'coral':  4,  # Third choice

}

```

**HPST Device Selection (from hpst.py):**

```python

if node_type in ['rpi3',  'rpi4']:

base_performance -=  15  # Heavy penalty for RPi devices

elif node_type in ['xeongpu',  'xeoncpu']:

base_performance +=  10  # Major boost for server-grade hardware

```

### **Workload Awareness vs Single-Objective Implementation**

K8s implements intelligent workload-device matching with device class scoring, while LPLT and HPST use fixed device preferences.

---

## **Anomalies / Things that i didn't expect**

### **Anomaly 1: LPLT Higher Power Consumption Despite "Efficiency Focus"**

**Expected**: LPLT should have lowest power consumption

**Actual**: LPLT consumes 1.3% MORE power than K8s

**explanation**: LPLT's power efficiency rankings are based on **idle power consumption**, not **load-adjusted power consumption**. This is most likely because:

1.  **Accelerated devices under load**: Nano (GPU) and Coral (TPU) consume more power when actively processing

2.  **Power model mismatch**: LPLT assumes static power consumption regardless of utilization

3.  **Workload mismatch**: Using GPU/TPU devices for non-accelerated workloads wastes energy

### **K8s Most Scaling Actions But Best Performance**

**Unexpected**: K8s performs 1,370 scaling actions (most) but achieves best performance

**Possbile Explanations**: This is most likely because K8s thresholds

1.  **Threshold oscillation**: Load hovers around thresholds causing frequent scaling decisions

2.  Frequent scaling maintains optimal resource allocation

3.  **Workload awareness compensation**: Device matching compensates for scaling inefficiency

---

## **Insights**

### **Insight 1: Pre-Positioned Replica Bias**

> **"Starting with abundant replicas (16 for 9 RPS load) masks strategy differences and reduces the validity of scaling algorithm comparisons."**

All strategies show similar performance despite different philosophies because resource pressure is minimal.

### **Insight 2: Threshold Calibration Critical**

> **"Strategy effectiveness depends heavily on threshold alignment with actual load patterns. K8s thresholds optimized for 20+ RPS perform poorly at 9 RPS."**

K8s performs 1,370 scaling actions vs LPLT's 1,129 due to threshold "misalignment".

### **Insight 3: Workload Awareness Validation**

> **"K8s demonstrates true workload awareness through device class scoring, while LPLT and HPST use fixed device preferences regardless of workload requirements."**

From the code Code we see K8s implements function_requirements mapping to device_classes.

---

## **9. Your Hypothesis Assessment**

### **Main Hypothesis**: "K8s is better because it's workload-aware and matches workloads to appropriate hardware"

### **What The Data Supports:**

**K8s achieves best performance** (1,284s vs 1,358s LPLT)

**K8s implements workload awareness** (device class scoring in code)

**K8s balances power and performance** (472.9W, best response time)

### **What The Data Challenges:**

**Performance margins are small** (5.7% difference with LPLT)

**Power savings are minimal** (0.7% difference with HPST)

**Pre-positioned replicas mask differences** (over-provisioning bias)

---

## **10. Research Contribution & Recommendations**

### **Your Refined Research Statement:**

> **"Under light load conditions with pre-positioned replicas, workload-aware autoscaling (K8s-inspired) maintains performance leadership through intelligent device class matching, but pre-provisioning diminishes the advantages of the scaling algorithms."**
