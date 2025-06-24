# Understanding the Simulator for Energy Modeling: Phase 1 Roadmap

Here's a prioritized roadmap to understand the FaaS simulator structure for implementing energy modeling:

## 1. Core Resource Handling (High Priority)

- [x] **Resource State & Tracking**

  - [ ] Study resource.py: How resources are tracked per node and function
  - [ ] Understand `ResourceUtilization`, `NodeResourceUtilization`, and `ResourceState` classes
  - [ ] Focus on `put_resource` and `remove_resource` methods - they'll be critical for energy tracking

- [x] **Resource Monitoring**
  - [ ] Examine resource.py: `ResourceMonitor` class
  - [ ] Study `run()` method - this is where periodic resource sampling happens
  - [ ] Note how `env.metrics.log_function_resource_utilization()` is called - you'll extend this pattern

## 2. Simulator Environment (High Priority)

- [x] **Environment Setup**

  - [ ] Study core.py: `Environment` class structure
  - [ ] Note how `power_models` dictionary was added - you'll use similar approach
  - [ ] Understand how the environment connects other components

- [x] **Metrics Collection System**
  - [ ] Examine metrics.py: How metrics are logged and extracted
  - [ ] Look for `log_function_resource_utilization` method
  - [ ] Study how metrics are extracted to DataFrames in main.py (lines ~99-110)

## 3. Function Execution Flow (Medium Priority)

- [x] **Function Simulators**

  - [ ] Study functionsim.py: How functions are simulated
  - [ ] Focus on `claim_resources` and `invoke` methods in `AIPythonHTTPSimulator` class
  - [ ] Understand how execution time is sampled and resources are consumed

- [x] **Simulation Initialization**
  - [ ] Look at main.py lines 77-93: How environment is initialized
  - [ ] Note how `simulator_factory` is set and metrics are configured

## 4. Device & Resource Characterization (Medium Priority)

- [x] **Device Models**

  - [ ] Examine resources.py: Resource profiles for different devices
  - [ ] Look at model.py: Device characteristic definitions
  - [ ] Study the `FunctionResourceCharacterization` format (CPU, GPU, IO, network, RAM)

- [x] **Function Execution Time Oracles**
  - [ ] Review fet.py: Execution time distributions
  - [ ] Examine oracles.py: How predictions are provided

## 5. Running and Analyzing Simulation (Medium Priority)

- [x] **Simulation Execution**

  - [ ] Study faassim.py: How simulations are run
  - [ ] Understand how background processes are started (like `ResourceMonitor`)

- [x] **Results Analysis**
  - [ ] Look at main.py lines 99-110: How metrics are extracted
  - [ ] Check extract.py: Analysis helpers

## 6. Extra Reference for Energy Implementation

- [x] **Power Prediction Example**
  - [ ] Review functionsim.py: Power prediction implementation
  - [ ] Study `PowerPredictionSimulator` class, especially `claim_resources` method
  - [ ] Understand how ML models are used for power prediction

## Files checklist in Order:

1. resource.py - Resource tracking foundation
2. core.py - Environment structure
3. functionsim.py - Function execution simulation
4. resources.py - Resource profiles
5. metrics.py - Metrics collection system
6. functionsim.py - Power prediction reference
