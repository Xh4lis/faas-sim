# Thorough Explanation: Deployments and Function Images in Raith21

Let's dive deep into these two critical lines that set up the function deployments and container images in the Raith21 simulation:

```python
deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())
function_images = images.all_ai_images
```

## 1. Creating Function Deployments

```python
deployments = list(create_all_deployments(fet_oracle, resource_oracle).values())
```

This line creates a list of deployment objects that define how serverless functions should be deployed in the simulated environment.

### What is `create_all_deployments`?

The `create_all_deployments` function is defined in deployments.py and creates deployment specifications for all the AI functions in the simulation. 

It:

1. **Takes two oracles as input**:
   - `fet_oracle`: Provides function execution time predictions
   - `resource_oracle`: Determines resource requirements

2. **Returns a dictionary** where:
   - Keys are function names (e.g., 'resnet50-inference')
   - Values are `FunctionDeployment` objects

3. **Each `FunctionDeployment` object contains**:
   - Function name and ID
   - Container image to use
   - Resource requirements (CPU, memory, etc.)
   - Scaling configuration
   - Scheduling constraints

### How the `deployments` Line Works:

1. `create_all_deployments(fet_oracle, resource_oracle)` returns a dictionary of function deployments
2. `.values()` extracts just the `FunctionDeployment` objects (not the function names)
3. `list(...)` converts the dictionary values to a list
4. The resulting `deployments` variable is a list of all function deployment specifications

### Example of What Gets Created:

A typical deployment might look like:
```python
FunctionDeployment(
    name='resnet50-inference',
    image='faas-workloads/resnet-inference-gpu',
    scaling_config=ScalingConfiguration(scale_min=1, scale_max=10),
    node_selector={'gpu': 'true'},  # Target nodes with GPUs
    resource_requirements=ResourceRequirements(
        cpu=2,       # CPU cores
        memory=4096  # MB of RAM
    )
)
```

## 2. Setting Function Images

```python
function_images = images.all_ai_images
```

This line sets up the container images that will be used for the functions in the simulation.

### What is `images.all_ai_images`?

The `all_ai_images` variable is defined in images.py and contains a list of container image definitions that represent AI workloads.

### Structure of `all_ai_images`:

It's a list of `FunctionImage` objects, where each object contains:
- Image name (e.g., 'faas-workloads/resnet-inference-gpu')
- Image properties (size, supported architectures)
- Tags (e.g., 'latest')

### Example of What `function_images` Contains:

```python
[
    FunctionImage(
        name='faas-workloads/resnet-inference-gpu',
        properties=[
            ImageProperties(name='faas-workloads/resnet-inference-gpu', size=2000000000, tag='latest', arch='x86'),
            ImageProperties(name='faas-workloads/resnet-inference-gpu', size=2000000000, tag='latest', arch='amd64'),
            ImageProperties(name='faas-workloads/resnet-inference-gpu', size=1000000000, tag='latest', arch='aarch64')
        ]
    ),
    FunctionImage(
        name='faas-workloads/resnet-inference-cpu',
        properties=[
            # Different variants for different architectures
            ImageProperties(name='faas-workloads/resnet-inference-cpu', size=2000000000, tag='latest', arch='x86'),
            ImageProperties(name='faas-workloads/resnet-inference-cpu', size=700000000, tag='latest', arch='arm32')
            # ... more architecture variants ...
        ]
    ),
    # ... more function images ...
]
```

## The Relationship Between These Two Lines

Together, these lines establish a complete blueprint for function deployment in the simulation:

1. **The `deployments` list** defines *what* functions will be deployed and their configuration
   - Includes names, scaling settings, and resource requirements
   - References image names (e.g., 'faas-workloads/resnet-inference-gpu')

2. **The `function_images` list** defines the *container images* that will be used
   - Specifies image properties like size and architecture
   - Enables architecture-aware image selection during node assignment

3. **Connection point**: When the simulation deploys a function from the `deployments` list, it will pull the appropriate container image variant from `function_images` based on:
   - The image name specified in the deployment
   - The architecture of the target node
   - Available image variants

## How This Impacts the Simulation

These two lines are crucial because they:

1. **Enable heterogeneous computing**: Different function variants can be deployed on different hardware architectures
2. **Model realistic constraints**: Images have different sizes, which affects pull times and storage requirements
3. **Support hardware-aware scheduling**: Functions can be placed on appropriate hardware based on their image requirements
4. **Allow for auto-scaling**: The deployment configurations include scaling parameters
5. **Define the actual workload mix**: These deployments are what will be invoked by the benchmark

When the simulation runs, these deployments will be scheduled onto the simulated nodes based on resource availability, hardware compatibility, and scheduling policies. The container images will be "pulled" to nodes (simulating network transfer of image data) before functions can execute.