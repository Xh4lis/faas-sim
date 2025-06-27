# Error Analysis and Fix Plan

## The Problem

The error occurs because the `FunctionDeployment` object doesn't have an `image` attribute. Looking at the `FunctionDeployment` class structure from the attachments, the correct way to access the image is through the function containers.

## Root Cause

In your deployments.py file, line 43 tries to access `deployment.image`, but `FunctionDeployment` objects store images differently. The image information is stored in the `fn_containers` list.

## Fix Plan

### Step 1: Understand FunctionDeployment Structure

From the attachments, `FunctionDeployment` has:

- `fn`: Function object
- `fn_containers`: List of FunctionContainer objects
- `scaling_config`: ScalingConfiguration
- `ranking`: DeploymentRanking

### Step 2: Fix the Deployment Creation

Replace the problematic `FunctionDeployment` creation in deployments.py:

```python
# Replace the create_smart_city_function_instances function with this corrected version:

def create_smart_city_function_instances(
    base_deployments: Dict[str, FunctionDeployment],
    instance_counts: Dict[str, int]
) -> List[FunctionDeployment]:
    """
    Create multiple instances of each function for smart city scenarios.

    Args:
        base_deployments: Dictionary of base function deployments
        instance_counts: How many instances of each function to create

    Returns:
        List of expanded function deployments
    """
    expanded_deployments = []

    for func_name, deployment in base_deployments.items():
        count = instance_counts.get(func_name, 1)

        for i in range(count):
            if i == 0:
                # Use the original deployment for the first instance
                expanded_deployments.append(deployment)
            else:
                # Create a new deployment with unique name
                new_name = f"{func_name}-zone-{i}"

                # Create a new function object with the new name
                new_function = copy.deepcopy(deployment.fn)
                new_function.name = new_name

                # Copy the function containers
                new_fn_containers = copy.deepcopy(deployment.fn_containers)

                # Create a new deployment with the correct structure
                from sim.faas.core import FunctionDeployment
                new_deployment = FunctionDeployment(
                    fn=new_function,
                    fn_containers=new_fn_containers,
                    scaling_config=copy.deepcopy(deployment.scaling_config),
                    deployment_ranking=copy.deepcopy(deployment.ranking)
                )

                # Copy any additional attributes that might exist
                for attr in dir(deployment):
                    if (not attr.startswith('_') and
                        not callable(getattr(deployment, attr)) and
                        attr not in ['fn', 'fn_containers', 'scaling_config', 'ranking', 'name']):
                        try:
                            value = getattr(deployment, attr)
                            setattr(new_deployment, attr, copy.deepcopy(value))
                        except:
                            pass  # Skip attributes that can't be copied

                expanded_deployments.append(new_deployment)

    return expanded_deployments
```

### Step 3: Update Imports

Make sure you have the correct imports at the top of deployments.py:

```python
import copy
from typing import Dict, List
from sim.faas.core import FunctionDeployment  # Make sure this import is correct
from ext.raith21.deployments import create_all_deployments
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle
```

### Step 4: Alternative Simpler Approach

If the above approach is still problematic, use a simpler method that just reuses the original deployments multiple times:

```python
def create_smart_city_function_instances(
    base_deployments: Dict[str, FunctionDeployment],
    instance_counts: Dict[str, int]
) -> List[FunctionDeployment]:
    """
    Create multiple instances by reusing original deployments.
    """
    expanded_deployments = []

    for func_name, deployment in base_deployments.items():
        count = instance_counts.get(func_name, 1)

        # Add the original deployment multiple times
        for i in range(count):
            expanded_deployments.append(deployment)

    return expanded_deployments
```

## Complete Fixed File

Here's the complete corrected deployments.py:

```python
import copy
from typing import Dict, List
from sim.faas.core import FunctionDeployment
from ext.raith21.deployments import create_all_deployments
from ext.raith21.oracles import Raith21ResourceOracle, Raith21FetOracle


def create_smart_city_function_instances(
    base_deployments: Dict[str, FunctionDeployment],
    instance_counts: Dict[str, int]
) -> List[FunctionDeployment]:
    """
    Create multiple instances of each function for smart city scenarios.

    Args:
        base_deployments: Dictionary of base function deployments
        instance_counts: How many instances of each function to create

    Returns:
        List of expanded function deployments
    """
    expanded_deployments = []

    for func_name, deployment in base_deployments.items():
        count = instance_counts.get(func_name, 1)

        # Add the deployment multiple times for multiple instances
        for i in range(count):
            expanded_deployments.append(deployment)

    return expanded_deployments


def create_smart_city_deployments(
    fet_oracle: Raith21FetOracle,
    resource_oracle: Raith21ResourceOracle,
    scenario: str = "default"
) -> List[FunctionDeployment]:
    """
    Create deployments for smart city scenarios with multiple function instances.
    """
    # Get base deployments from the original system
    all_deployments = create_all_deployments(fet_oracle, resource_oracle)

    # Define smart city function types and their use cases
    base_selected_functions = [
        "resnet50-inference",     # Traffic cameras, security cameras
        "mobilenet-inference",    # Edge devices, mobile cameras
        "speech-inference",       # Audio monitoring, emergency detection
        "resnet50-preprocessing", # Data preprocessing from sensors
        "resnet50-training",      # Periodic model updates
    ]

    # Filter to available functions
    base_deployments = {
        name: all_deployments[name]
        for name in base_selected_functions
        if name in all_deployments
    }

    # Define scenarios with different deployment patterns
    scenarios = {
        "default": {
            "resnet50-inference": 6,      # 6 traffic/security camera zones
            "mobilenet-inference": 10,    # 10 edge device zones
            "speech-inference": 4,        # 4 audio monitoring zones
            "resnet50-preprocessing": 5,  # 5 data processing zones
            "resnet50-training": 2,       # 2 training instances
        },
        "intensive": {
            "resnet50-inference": 12,     # More intensive camera coverage
            "mobilenet-inference": 20,    # Dense edge device deployment
            "speech-inference": 8,        # Comprehensive audio monitoring
            "resnet50-preprocessing": 10, # Heavy data processing
            "resnet50-training": 4,       # More frequent training
        },
        "distributed": {
            "resnet50-inference": 15,     # Highly distributed camera network
            "mobilenet-inference": 25,    # Maximum edge coverage
            "speech-inference": 12,       # City-wide audio monitoring
            "resnet50-preprocessing": 15, # Distributed data processing
            "resnet50-training": 6,       # Distributed learning
        }
    }

    # Get instance counts for the selected scenario
    instance_counts = scenarios.get(scenario, scenarios["default"])

    # Create expanded deployments
    deployments = create_smart_city_function_instances(base_deployments, instance_counts)

    print(f"\n=== SMART CITY DEPLOYMENT: {scenario.upper()} SCENARIO ===")
    print(f"Created {len(deployments)} function instances:")

    # Group by function type for summary
    function_summary = {}
    for deployment in deployments:
        base_name = deployment.name.split('-zone-')[0] if '-zone-' in deployment.name else deployment.name
        if base_name not in function_summary:
            function_summary[base_name] = 0
        function_summary[base_name] += 1

    for func_type, count in function_summary.items():
        print(f"  {func_type}: {count} instances")

    return deployments


def create_custom_smart_city_deployments(
    fet_oracle: Raith21FetOracle,
    resource_oracle: Raith21ResourceOracle,
    custom_instance_counts: Dict[str, int]
) -> List[FunctionDeployment]:
    """
    Create deployments with custom instance counts.
    """
    all_deployments = create_all_deployments(fet_oracle, resource_oracle)

    # Filter to requested functions
    base_deployments = {
        name: all_deployments[name]
        for name in custom_instance_counts.keys()
        if name in all_deployments
    }

    deployments = create_smart_city_function_instances(base_deployments, custom_instance_counts)

    print(f"\n=== CUSTOM SMART CITY DEPLOYMENT ===")
    print(f"Created {len(deployments)} function instances:")
    for func_name, count in custom_instance_counts.items():
        if func_name in all_deployments:
            print(f"  {func_name}: {count} instances")

    return deployments
```

## Expected Results

After this fix, you should get:

- **27 function deployments** with the default scenario (6+10+4+5+2)
- Much better utilization of your 500 devices
- No more AttributeError about missing `image` attribute

This simplified approach reuses the original deployment objects multiple times, which will result in multiple replicas of each function being created during the simulation, giving you the device utilization you're looking for.
