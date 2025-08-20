import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class Raith21PowerOracle:
    """
    Linear power model: P_total = P_idle + (CPU_util * P_cpu_max) + (GPU_util * P_gpu_max) + (Network_util * P_network_max)
    """
    
    def __init__(self, power_profiles: Dict[str, Dict[str, float]]):
        self.power_profiles = power_profiles
        
    def predict_power(self, node_type: str, cpu_util: float, gpu_util: float = 0.0, 
                     network_util: float = 0.0, memory_util: float = 0.0) -> float:
        """
        Calculate power consumption using linear model
        
        Args:
            node_type: Device type ('rpi4', 'nano', 'xeongpu')
            cpu_util: CPU utilization (0.0-1.0)
            gpu_util: GPU utilization (0.0-1.0) 
            network_util: Network utilization (0.0-1.0)
            memory_util: Memory utilization (0.0-1.0)
            
        Returns:
            Power consumption in Watts
        """
        profile = self.power_profiles.get(node_type)
        if not profile:
            logger.warning(f"No power profile for {node_type}, using default")
            profile = self.power_profiles.get("default", {
                "idle": 5.0, "cpu_max": 10.0, "gpu_max": 15.0, 
                "network_max": 2.0, "memory_max": 1.0
            })
        
        # Linear power model
        power = profile["idle"]
        power += cpu_util * profile.get("cpu_max", 0.0)
        power += gpu_util * profile.get("gpu_max", 0.0)
        power += network_util * profile.get("network_max", 0.0)
        power += memory_util * profile.get("memory_max", 0.0)
        
        return power

# Device-specific power profiles based on
DEVICE_POWER_PROFILES = {
    # Edge Devices
    "rpi3": {
        "idle": 2.1,      
        "cpu_max": 1.9,  
        "gpu_max": 0.5,   
        "network_max": 0.8,
        "memory_max": 0.4
    },
    "rpi4": {
        "idle": 2.7,      
        "cpu_max": 3.8,   
        "gpu_max": 1.0,  
        "network_max": 1.0,
        "memory_max": 0.5
    },
    
    # NVIDIA Jetson Devices
    "nano": {
        "idle": 1.9,      
        "cpu_max": 3.1,   
        "gpu_max": 2.0,   
        "network_max": 0.5,
        "memory_max": 0.6
    },
    "tx2": {
        "idle": 5.5,      
        "cpu_max": 6.5,
        "gpu_max": 12.0,  
        "network_max": 1.5,
        "memory_max": 1.0
    },
    "nx": {
        "idle": 5.0,      
        "cpu_max": 10.0,
        "gpu_max": 15.0,  
        "network_max": 2.0,
        "memory_max": 1.5
    },
    
    # Other Edge Devices
    "coral": {
        "idle": 2.5,      
        "cpu_max": 2.5,
        "gpu_max": 0.0,   
        "tpu_max": 4.0,   
        "network_max": 0.6,
        "memory_max": 0.3
    },
    "rockpi": {
        "idle": 3.0,      
        "cpu_max": 4.0,
        "gpu_max": 1.5,
        "network_max": 0.8,
        "memory_max": 0.6
    },
    
    # Server Grade
    "nuc": {
        "idle": 15.0,     
        "cpu_max": 50.0,  
        "gpu_max": 0.0,  
        "network_max": 3.0,
        "memory_max": 2.0
    },
    "xeoncpu": {
        "idle": 50.0,     
        "cpu_max": 200.0, 
        "gpu_max": 0.0,
        "network_max": 8.0,
        "memory_max": 5.0
    },
    "xeongpu": {
        "idle": 75.0,     
        "cpu_max": 205.0, 
        "gpu_max": 275.0, 
        "network_max": 10.0,
        "memory_max": 8.0
    },
    
    # Default fallback
    "default": {
        "idle": 5.0,
        "cpu_max": 10.0,
        "gpu_max": 15.0,
        "network_max": 2.0,
        "memory_max": 1.0
    }
}


def extract_node_type(node_name: str) -> str:
    """Extract device type from node name (e.g., 'rpi4_3' -> 'rpi4')"""
    if '_' in node_name:
        return node_name.split('_')[0]
    return node_name

def get_current_utilization(env, node_name: str) -> Dict[str, float]:
    """Get current resource utilization for a node"""
    
    # Skip registry and infrastructure nodes
    if (node_name == 'registry' or 'registry' in node_name.lower() or 
        'switch' in node_name.lower() or 'link' in node_name.lower()):
        return None
        
    if not hasattr(env, 'resource_state') or env.resource_state is None:
        return get_device_defaults(node_name)
    
    try:
        # This returns: List[Tuple[FunctionReplica, ResourceUtilization]]
        replicas_on_node = env.resource_state.list_resource_utilization(node_name)
        
        if replicas_on_node:
            total_cpu = 0.0
            total_memory = 0.0
            total_gpu = 0.0
            
            # Iterate through (replica, resource_utilization) tuples
            for replica, resource_util in replicas_on_node:
                if resource_util and hasattr(resource_util, 'list_resources'):
                    resources = resource_util.list_resources()  # Returns Dict[str, float]
                    total_cpu += resources.get('cpu', 0.0)
                    total_memory += resources.get('memory', 0.0)
                    total_gpu += resources.get('gpu', 0.0)
            
            result = {
                'cpu': min(1.0, total_cpu),
                'memory': min(1.0, total_memory),
                'gpu': min(1.0, total_gpu),
                'network': min(1.0, total_cpu * 0.3)  # Network estimate
            }
            
            # Only print if there are non-zero utilization values (HIT/success)
            if any(value > 0.0 for value in result.values()):
                print(f"✅ HIT: Non-zero utilization for {node_name}: {result}")
            
            return result
        else:
            return get_device_defaults(node_name)
            
    except Exception as e:
        logger.error(f"ResourceState lookup failed for {node_name}: {e}")
        return get_device_defaults(node_name)

def get_device_defaults(node_name: str) -> Dict[str, float]:
    """Get realistic default utilization based on device type"""
    node_type = extract_node_type(node_name)
    
    if 'xeon' in node_type:
        return {'cpu': 0.15, 'gpu': 0.05, 'network': 0.08, 'memory': 0.12}
    elif node_type in ['nx', 'tx2']:
        return {'cpu': 0.25, 'gpu': 0.15, 'network': 0.10, 'memory': 0.18}
    elif node_type in ['nano', 'rpi4']:
        return {'cpu': 0.20, 'gpu': 0.08, 'network': 0.06, 'memory': 0.15}
    else:
        return {'cpu': 0.12, 'gpu': 0.02, 'network': 0.05, 'memory': 0.10}

def monitor_power_consumption(env):
    """Monitor power consumption across all nodes"""
    try:
        if not env.power_oracle or not env.power_metrics:
            return
            
        current_time = env.now
        
        # Monitor all nodes in the topology
        for node in env.topology.get_nodes():
            if hasattr(node, 'capacity'):  # Only monitor compute nodes
                if node.name == 'registry' or 'registry' in node.name.lower():
                    continue
                node_name = node.name
                node_type = extract_node_type(node_name)
                
                # Get current utilization
                utilization = get_current_utilization(env, node_name)
                
                # Calculate power consumption
                power_watts = env.power_oracle.predict_power(
                    node_type,
                    utilization['cpu'],
                    utilization['gpu'], 
                    utilization['network'],
                    utilization['memory']
                )
                
                # Record the measurement
                sample = env.power_metrics.record_power_sample(
                    current_time, node_name, node_type,
                    utilization['cpu'], utilization['gpu'],
                    utilization['network'], utilization['memory'],
                    power_watts
                )
                
                # Log to main metrics system for DataFrame extraction
                env.metrics.log('power', {
                    'timestamp': current_time,
                    'node': node_name,
                    'node_type': node_type,
                    'power_watts': power_watts,
                    'cpu_util': utilization['cpu'],
                    'gpu_util': utilization['gpu'],
                    'network_util': utilization['network'],
                    'memory_util': utilization['memory']
                })
                
                env.metrics.log('energy', {
                    'timestamp': current_time,
                    'node': node_name,
                    'energy_joules': sample['cumulative_energy_joules'],
                    'energy_wh': sample['cumulative_energy_wh']
                })
                
    except Exception as e:
        logger.error(f"Error monitoring power consumption: {e}")


def power_monitoring_loop(env):
    """SimPy process that continuously monitors power consumption"""
    logger.info("Power monitoring loop started")
    
    while True:
        try:
            # Wait for the monitoring interval
            yield env.timeout(env.power_monitoring_interval)
            
            # Perform power monitoring
            monitor_power_consumption(env)
            
        except Exception as e:
            logger.error(f"Error in power monitoring loop: {e}")
            # Continue monitoring even if there's an error
            yield env.timeout(1.0)  # Wait 1 second before retrying

def test_power_calculations():
    """Test power calculation functionality"""
    oracle = Raith21PowerOracle(DEVICE_POWER_PROFILES)
    
    # Test different scenarios
    test_cases = [
        ("rpi4", 0.0, 0.0, 0.0, 0.0),    # Idle
        ("rpi4", 0.5, 0.0, 0.1, 0.2),    # Light load
        ("rpi4", 1.0, 0.0, 0.3, 0.8),    # High load
        ("nx", 0.7, 0.9, 0.2, 0.6),      # GPU intensive
        ("xeongpu", 0.8, 0.6, 0.4, 0.9), # Server load
    ]
    
    print("🔋 Power Model Test Results:")
    print("Device\tCPU\tGPU\tNet\tMem\tPower(W)")
    print("-" * 45)
    
    for device, cpu, gpu, net, mem in test_cases:
        power = oracle.predict_power(device, cpu, gpu, net, mem)
        print(f"{device}\t{cpu:.1f}\t{gpu:.1f}\t{net:.1f}\t{mem:.1f}\t{power:.1f}W")

if __name__ == "__main__":
    test_power_calculations()
