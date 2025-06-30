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
    Create multiple UNIQUE instances of each function for smart city scenarios.
    """
    expanded_deployments = []
    
    for func_name, base_deployment in base_deployments.items():
        count = instance_counts.get(func_name, 1)
        
        for i in range(count):
            # Create a completely new deployment with unique name
            import copy
            
            # Deep copy the base deployment
            new_deployment = copy.deepcopy(base_deployment)
            # Force each deployment to start with 1 replica
            new_deployment.scaling_config.scale_min = 1
            new_deployment.scaling_config.scale_max = 3  

            # Create unique names for smart city zones
            zone_names = [
                "downtown", "suburb", "industrial", "residential", "commercial",
                "airport", "port", "university", "hospital", "mall",
                "stadium", "park", "transit", "highway", "border"
            ]
            
            if i == 0:
                # Keep original for first instance
                expanded_deployments.append(new_deployment)
            else:
                # Create unique deployment for each zone
                zone_name = zone_names[i-1] if i-1 < len(zone_names) else f"zone{i}"
                new_name = f"{func_name}-{zone_name}"
                
                # Update the function name to make it unique
                new_deployment.fn.name = new_name
                
                expanded_deployments.append(new_deployment)
    
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