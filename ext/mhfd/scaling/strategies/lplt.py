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
    - Optimizes total energy = low_power x long_time
    - Suitable for battery-powered edge deployments
    """
    
    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, "LowPowerLongTimeBinPacker")

        self.scale_up_threshold = 8
        self.scale_down_threshold = 3
        self.response_time_threshold = 1200
        # self.last_scale_action = {}  # Track last scaling action per deployment
        # self.scale_down_cooldown = 45 

        # Consolidation parameters
        self.target_utilization = 0.75      
        self.min_replicas_per_node = 2      
        
        # Define power efficiency rankings (watts idle power)
        self.power_efficiency_ranking = {
            'rpi3': 1,      # 1.4W - Lowest power, ensures LPLT hypothesis
                            # Source: RasPi.TV calibrated measurements
                            # Research Value: TRUE low power, low performance
            
            'nano': 2,      # 2.0W - Low power BUT GPU acceleration  
                            # Source: NVIDIA official 5W mode specifications
                            # Research Problem: Breaks LPLT hypothesis (too fast for AI)
            
            'rockpi': 3,    # 2.0W - CPU-focused, limited acceleration
                            # Source: Community power measurements
                            # Research Value: Reasonable LPLT device
            
            'coral': 4,     # 2.5W - Low power BUT TPU acceleration
                            # Source: Google official specifications
                            # Research Problem: Breaks LPLT hypothesis (too fast for AI)
            
            'rpi4': 5,      # 2.85W - Low power, moderate CPU performance
                            # Source: RasPi.TV calibrated measurements  
                            # Research Value: Good LPLT device for most workloads
            
            'tx2': 6,       # 5.0W - Moderate power, has GPU
                            # Source: NVIDIA efficiency mode specifications
                            # Research Transition: Between LPLT and HPST
            
            'nx': 7,        # 7.3W - Higher power, powerful GPU
                            # Source: Independent benchmark measurements
                            # Research Value: Good HPST device
            
            'nuc': 8,       # 8.0W - Consistently good general performance
                            # Source: Multiple review sites average
                            # Research Value: Excellent HPST device
            
            'xeoncpu': 9,   # 25.0W - Server-grade CPU performance
                            # Source: Server system estimates
                            # Research Value: High-end HPST device
            
            'xeongpu': 10   # 40.0W - Highest power, best absolute performance
                            # Source: Server + discrete GPU estimates
                            # Research Value: Maximum HPST performance
                }
    
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                            current_load: float, avg_response_time: float) -> str:
        
        current_time = self.env.now
        # last_action_time = self.last_scale_action.get(deployment_name, 0)
        
        logger.debug(f"🔋 LPLT {deployment_name}: Load={current_load:.1f}, "
                    f"RT={avg_response_time:.1f}ms, Replicas={current_replicas}")
        
        # SCALE UP: Energy-aware but responsive 
        if ((current_load > self.scale_up_threshold) or 
            (avg_response_time > self.response_time_threshold)) and \
           current_replicas < self.max_replicas:
            
            # self.last_scale_action[deployment_name] = current_time  
            logger.info(f"🔋 LPLT SCALE UP {deployment_name}: Load or RT threshold exceeded")
            return "scale_up"
        
        # SCALE DOWN: More aggressive for energy savings
        # time_since_last_action = current_time - last_action_time
        
        if (current_load < self.scale_down_threshold and 
            avg_response_time < (self.response_time_threshold * 0.5) and
            current_replicas > self.min_replicas):
            # time_since_last_action > self.scale_down_cooldown):  
            
            # self.last_scale_action[deployment_name] = current_time  
            logger.info(f"🔋 LPLT SCALE DOWN {deployment_name}: Load={current_load:.1f}, RT={avg_response_time:.1f}")
            return "scale_down"
        
        return "no_action"
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """Energy-aware with consolidation preference"""
        available_nodes = self.get_available_nodes()
        
        # 1. Try to consolidate on existing active low-power nodes
        active_low_power_nodes = self.get_active_low_power_nodes(available_nodes)
        for node in active_low_power_nodes:
            if self.can_consolidate_on_node(node, deployment_name):
                logger.info(f"🔋 LPLT Consolidating on {node.name}")
                return node
        
        # 2. If no consolidation possible, use workload-specific selection
        return self.select_new_low_power_node(available_nodes, deployment_name)
    
    def get_active_low_power_nodes(self, available_nodes) -> List[Any]:
        """Get low-power nodes that are already running workloads"""
        active_nodes = []
        low_power_types = ['rpi3', 'rpi4', 'nano', 'coral', 'rockpi']
        
        for node in available_nodes:
            node_type = self.extract_node_type(node.name)
            if node_type in low_power_types:
                utilization = self.get_node_utilization(node)
                if utilization.get('cpu', 0) > 0.1:  # Already active
                    active_nodes.append(node)
        
        # Sort by current utilization (prefer nodes with medium load)
        active_nodes.sort(key=lambda n: self.get_node_utilization(n).get('cpu', 0))
        return active_nodes
    
    def can_consolidate_on_node(self, node, deployment_name: str) -> bool:
        """Check if we can add more load to this node for energy efficiency"""
        utilization = self.get_node_utilization(node)
        
        # Allow higher consolidation for energy savings
        cpu_can_handle = utilization.get('cpu', 0) < self.target_utilization
        memory_can_handle = utilization.get('memory', 0) < self.target_utilization
        
        return cpu_can_handle and memory_can_handle
    
    def select_new_low_power_node(self, available_nodes, deployment_name: str) -> Optional[Any]:
        """Select new low-power node based on workload type"""
        workload_preferences = {
            'inference': ['coral', 'nano', 'rpi4', 'rpi3'],  # TPU/GPU first
            'training': ['nano', 'rpi4', 'rockpi', 'rpi3'],  # GPU/CPU capable
            'cpu': ['rpi4', 'rockpi', 'rpi3', 'nano']        # CPU-focused
        }
        
        # Determine workload type
        if 'inference' in deployment_name.lower():
            preferred_types = workload_preferences['inference']
        elif 'training' in deployment_name.lower():
            preferred_types = workload_preferences['training']
        else:
            preferred_types = workload_preferences['cpu']
        
        # Try preferred types in order
        for node_type in preferred_types:
            for node in available_nodes:
                if (node_type in node.name.lower() and 
                    self.has_sufficient_resources(node, deployment_name)):
                    logger.info(f"🔋 LPLT New node: {node.name} ({node_type}) for {deployment_name}")
                    return node
        
        return None

    def has_sufficient_resources(self, node, deployment_name: str) -> bool:
        """Balanced resource checking"""
        try:
            utilization = self.get_node_utilization(node)
            
            # Balanced thresholds - not too strict, not too lenient
            cpu_available = utilization.get('cpu', 0) < 0.8    # 80% threshold
            memory_available = utilization.get('memory', 0) < 0.8
            
            return cpu_available and memory_available
        except:
            return True
    
    
    def estimate_node_power_consumption(self, node: Any, deployment_name: str) -> float:
        """Estimate power consumption for running function on node using real utilization"""
        node_type = self.extract_node_type(node.name)
        utilization = self.get_node_utilization(node)
        cpu_util = utilization.get('cpu', 0.0)
        gpu_util = utilization.get('gpu', 0.0)
        network_util = utilization.get('network', 0.0)
        memory_util = utilization.get('memory', 0.0)

        return self.power_oracle.predict_power(
            node_type, cpu_util, gpu_util, network_util, memory_util
        )

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
            raise e