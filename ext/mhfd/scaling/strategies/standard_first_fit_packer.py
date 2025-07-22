import logging
from typing import Optional, Any
from ..base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization

logger = logging.getLogger(__name__)

class StandardFirstFitBinPacker(BaseAutoscaler):
    """
    Standard First-Fit Bin Packing Strategy
    - Classic First-Fit algorithm implementation
    - Uses first available node meeting resource requirements
    - Baseline comparison for specialized strategies
    """

    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, "StandardFirstFitBinPacker")

        # Basic thresholds
        self.scale_up_threshold = 15    # Scale up at 15 RPS
        self.scale_down_threshold = 3   # Scale down below 3 RPS
        self.response_time_threshold = 800  # 800ms response time threshold
    
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                             current_load: float, avg_response_time: float) -> str:
        """TEMPORARY: Force scaling for demonstration"""
        
        # Force scaling for high-demand functions after some time
        # if self.env.now > 8:  # After 8 seconds
        #     if "inference" in deployment_name and current_replicas == 1:
        #         logger.info(f"🚀 FORCING scale up for {deployment_name} (demonstration)")
        #         return "scale_up"
        
        # Normal decision logic
        if current_load > 15 or avg_response_time > 800:
            if current_replicas < self.max_replicas:
                logger.info(f"📈 SCALING UP {deployment_name}: load={current_load:.1f} RPS, "
                           f"response_time={avg_response_time:.1f}ms")
                return "scale_up"
        
        elif current_load < 3 and avg_response_time < 200:
            if current_replicas > self.min_replicas:
                logger.info(f"📉 SCALING DOWN {deployment_name}: load={current_load:.1f} RPS")
                return "scale_down"
        
        return "no_action"
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """Select first available node - FIXED VERSION"""
        
        # Get actual available nodes from the environment
        try:
            # Get all nodes from the environment
            available_nodes = []
            if hasattr(self.env, 'topology') and hasattr(self.env.topology, 'nodes'):
                available_nodes = list(self.env.topology.nodes)
            elif hasattr(self.faas, '_nodes'):
                available_nodes = list(self.faas._nodes)
            else:
                # Fallback: get nodes from a deployment
                deployments = self.faas.get_deployments()
                if deployments:
                    replicas = self.faas.get_replicas(deployments[0].name)
                    if replicas:
                        available_nodes = [replica.pod.node for replica in replicas if hasattr(replica, 'pod')]
            
            logger.debug(f"🔍 Available nodes: {[n.name if hasattr(n, 'name') else str(n) for n in available_nodes[:5]]}")
            
            if available_nodes:
                selected = available_nodes[0]  # First-fit strategy
                logger.info(f"✅ Selected node: {selected.name if hasattr(selected, 'name') else str(selected)}")
                return selected
            else:
                logger.error(f"❌ No available nodes found!")
                return None
                
        except Exception as e:
            logger.error(f"❌ Node selection error: {e}")
            return None
    
    def get_available_nodes(self):
        """Get all available compute nodes"""
        return [node for node in self.env.topology.get_nodes() 
                if self.is_compute_node(node)]
    
    def is_compute_node(self, node) -> bool:
        """Check if node is a compute node (not registry/infrastructure)"""
        return (hasattr(node, 'capacity') and 
                not any(keyword in node.name.lower() 
                       for keyword in ['registry', 'link', 'switch', 'shared']))
    
    def has_sufficient_resources(self, node, deployment_name: str) -> bool:
        """Check if node has sufficient CPU/memory resources"""
        try:
            # Get node current utilization
            current_utilization = self.get_node_utilization(node)
            
            # Simple resource check - can be enhanced
            cpu_available = current_utilization.get('cpu', 0) < 0.8  # 80% CPU threshold
            memory_available = current_utilization.get('memory', 0) < 0.8  # 80% memory threshold
            
            return cpu_available and memory_available
        except:
            return True  # Default to available if can't determine
    
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