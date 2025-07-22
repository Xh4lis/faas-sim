import logging
from typing import Optional, Any, List, Tuple
from ..base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization

logger = logging.getLogger(__name__)

class LowPowerLongTimeBinPacker(BaseAutoscaler):
    """
    Low-Power Long-Time Energy Optimization Strategy
    
    Energy optimization through device power efficiency:
    - Prioritizes nodes with lowest idle/active power consumption (RPi, Nano, Coral)
    - Accepts longer execution times for lower power consumption
    - Optimizes total energy = low_power × long_time
    - Suitable for battery-powered edge deployments
    """
    
    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, "LowPowerLongTimeBinPacker")

        # Power-aware thresholds (more conservative)
        self.scale_up_threshold = 10     # Scale up at 10 RPS (earlier)
        self.scale_down_threshold = 2    # Scale down below 2 RPS
        self.response_time_threshold = 1500  # Accept 1500ms response time
        self.max_power_per_node = 5.0    # Max 5W per node target
        
        # Define power efficiency rankings (watts idle power)
        self.power_efficiency_ranking = {
            'coral': 1,     # Most efficient (2.5W idle)
            'nano': 2,      # Second (1.9W idle) 
            'rpi3': 3,      # Third (2.1W idle)
            'rpi4': 4,      # Fourth (2.7W idle)
            'rockpi': 5,    # Fifth (3.0W idle)
            'tx2': 6,       # Sixth (5.5W idle)
            'nx': 7,        # Seventh (5.0W idle)
            'nuc': 8,       # Higher power (15W idle)
            'xeoncpu': 9,   # Much higher (50W idle)
            'xeongpu': 10   # Highest power (75W idle)
        }
    
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                            current_load: float, avg_response_time: float) -> str:
        """Power-aware scaling decision - prioritize energy efficiency"""
        
        # Calculate current power efficiency
        current_power_efficiency = self.calculate_power_efficiency(deployment_name)
        
        logger.debug(f"🔋 PowerOptimized: {deployment_name} - Load: {current_load:.1f} RPS, "
                    f"Response: {avg_response_time:.1f}ms, Power Efficiency: {current_power_efficiency:.2f} W/RPS")
        
        # Scale up conditions (more aggressive for power efficiency)
        if (current_load > self.scale_up_threshold or 
            avg_response_time > self.response_time_threshold or
            current_power_efficiency > 8.0) and \
           current_replicas < self.max_replicas:
            return "scale_up"
        
        # Scale down conditions (more conservative - keep low-power nodes)
        elif current_load < self.scale_down_threshold and \
             avg_response_time < (self.response_time_threshold * 0.3) and \
             current_replicas > self.min_replicas:
            return "scale_down"
        
        return "no_action"
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """Power-optimized: Select most energy-efficient available node"""
        
        available_nodes = self.get_available_nodes()
        
        # Sort nodes by power efficiency (lowest power first)
        power_ranked_nodes = self.rank_nodes_by_power_efficiency(available_nodes)
        
        for node, power_score in power_ranked_nodes:
            if self.has_sufficient_resources(node, deployment_name):
                estimated_power = self.estimate_node_power_consumption(node, deployment_name)
                
                logger.info(f"🔋 PowerOptimized: Selected {node.name} "
                           f"(estimated power: {estimated_power:.1f}W, efficiency rank: {power_score})")
                return node
        
        logger.warning("❌ PowerOptimized: No energy-efficient nodes available")
        return None
    
    def rank_nodes_by_power_efficiency(self, nodes: List[Any]) -> List[Tuple[Any, int]]:
        """Rank nodes by power efficiency (lower is better)"""
        ranked_nodes = []
        
        for node in nodes:
            node_type = self.extract_node_type(node.name)
            power_rank = self.power_efficiency_ranking.get(node_type, 99)  # Unknown = worst
            ranked_nodes.append((node, power_rank))
        
        # Sort by power rank (ascending - most efficient first)
        ranked_nodes.sort(key=lambda x: x[1])
        
        return ranked_nodes
    
    def estimate_node_power_consumption(self, node: Any, deployment_name: str) -> float:
        """Estimate power consumption for running function on node"""
        node_type = self.extract_node_type(node.name)
        
        # Estimate utilization for the workload
        estimated_cpu_util = 0.6  # Assume 60% CPU utilization
        estimated_gpu_util = 0.1 if 'gpu' in deployment_name.lower() else 0.0
        estimated_network_util = 0.2  # 20% network utilization
        estimated_memory_util = 0.4   # 40% memory utilization
        
        return self.power_oracle.predict_power(
            node_type, estimated_cpu_util, estimated_gpu_util,
            estimated_network_util, estimated_memory_util
        )
    
    def calculate_power_efficiency(self, deployment_name: str) -> float:
        """Calculate current power efficiency (Watts per RPS)"""
        try:
            replicas = self.faas.get_replicas(deployment_name)
            if not replicas:
                return 0.0
            
            total_power = 0.0
            current_load = self.get_current_load(deployment_name)
            
            for replica in replicas:
                node_name = replica.node.name
                node_type = self.extract_node_type(node_name)
                
                # Get current utilization
                utilization = self.get_node_utilization(replica.node)
                
                # Calculate power consumption
                power = self.power_oracle.predict_power(
                    node_type, utilization['cpu'], utilization['gpu'],
                    utilization['network'], utilization['memory']
                )
                total_power += power
            
            # Return watts per RPS (lower is better)
            return total_power / max(current_load, 0.1)
        except:
            return 5.0  # Default moderate efficiency
    
    def extract_node_type(self, node_name: str) -> str:
        """Extract node type from node name"""
        for node_type in self.power_efficiency_ranking.keys():
            if node_type in node_name.lower():
                return node_type
        return 'unknown'
    
    def get_available_nodes(self):
        """Get available compute nodes"""
        return [node for node in self.env.topology.get_nodes() 
                if self.is_compute_node(node)]
    
    def is_compute_node(self, node) -> bool:
        """Check if node is a compute node"""
        return (hasattr(node, 'capacity') and 
                not any(keyword in node.name.lower() 
                       for keyword in ['registry', 'link', 'switch', 'shared']))
    
    def has_sufficient_resources(self, node, deployment_name: str) -> bool:
        """Check if node has sufficient resources (more lenient for power efficiency)"""
        try:
            current_utilization = self.get_node_utilization(node)
            
            # More lenient resource thresholds for power efficiency
            cpu_available = current_utilization.get('cpu', 0) < 0.9  # 90% CPU threshold
            memory_available = current_utilization.get('memory', 0) < 0.85  # 85% memory threshold
            
            return cpu_available and memory_available
        except:
            return True


    def get_node_utilization(self, node) -> dict:
        """Get  current node resource utilization from ResourceState"""
        try:
            real_utilization = get_current_utilization(self.env, node.name)
            
            if real_utilization is None:
                logger.error(f"❌ Failed to get real utilization for {node.name}")
                return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}
            
            logger.debug(f"🔍 REAL utilization for {node.name}: {real_utilization}")
            return real_utilization
            
        except Exception as e:
            logger.error(f"❌ EXCEPTION Failed to get real utilization for {node.name}: {e}")
            raise  