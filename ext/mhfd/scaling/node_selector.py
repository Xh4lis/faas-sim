import logging
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class NodeSelector(ABC):
    """Base class for node selection strategies"""
    
    def __init__(self, env, power_oracle):
        self.env = env
        self.power_oracle = power_oracle
    
    @abstractmethod
    def select_best_node(self, deployment_name: str, available_nodes: List[Any]) -> Optional[Any]:
        """Select the best node for scaling based on strategy"""
        pass
    
    def get_available_nodes(self) -> List[Any]:
        """Get all available compute nodes"""
        return [node for node in self.env.topology.get_nodes() 
                if self.is_compute_node(node)]
    
    def is_compute_node(self, node) -> bool:
        """Check if node is a compute node (not infrastructure)"""
        return (hasattr(node, 'capacity') and 
                not any(keyword in node.name.lower() 
                       for keyword in ['registry', 'link', 'switch', 'shared']))
    
    def has_sufficient_resources(self, node, deployment_name: str, 
                               cpu_threshold: float = 0.8, 
                               memory_threshold: float = 0.8) -> bool:
        """Check if node has sufficient resources"""
        try:
            from ext.mhfd.power import get_current_utilization
            
            current_utilization = get_current_utilization(self.env, node.name)
            
            cpu_available = current_utilization.get('cpu', 0) < cpu_threshold
            memory_available = current_utilization.get('memory', 0) < memory_threshold
            
            return cpu_available and memory_available
        except:
            return True  # Default to available if can't determine
    
    def extract_node_type(self, node_name: str) -> str:
        """Extract node type from node name"""
        node_types = ['coral', 'nano', 'rpi3', 'rpi4', 'rockpi', 'tx2', 'nx', 
                     'nuc', 'xeoncpu', 'xeongpu', 'registry']
        
        for node_type in node_types:
            if node_type in node_name.lower():
                return node_type
        return 'unknown'


class FirstFitNodeSelector(NodeSelector):
    """Selects first available node with sufficient resources"""
    
    def select_best_node(self, deployment_name: str, available_nodes: List[Any]) -> Optional[Any]:
        """Select first available node"""
        for node in available_nodes:
            if self.has_sufficient_resources(node, deployment_name):
                logger.debug(f"✅ FirstFit selected: {node.name}")
                return node
        
        logger.warning("❌ FirstFit: No available nodes")
        return None


class PowerOptimizedNodeSelector(NodeSelector):
    """Selects most energy-efficient node"""
    
    def __init__(self, env, power_oracle):
        super().__init__(env, power_oracle)
        
        # Power efficiency ranking (watts idle power - lower is better)
        self.power_efficiency_ranking = {
            'coral': 2.5,     # Most efficient
            'nano': 1.9,      
            'rpi3': 2.1,      
            'rpi4': 2.7,      
            'rockpi': 3.0,    
            'tx2': 5.5,       
            'nx': 5.0,        
            'nuc': 15.0,      
            'xeoncpu': 50.0,  
            'xeongpu': 75.0   # Least efficient
        }
    
    def select_best_node(self, deployment_name: str, available_nodes: List[Any]) -> Optional[Any]:
        """Select most energy-efficient available node"""
        
        # Filter nodes with sufficient resources
        suitable_nodes = [node for node in available_nodes 
                         if self.has_sufficient_resources(node, deployment_name, 
                                                        cpu_threshold=0.9, 
                                                        memory_threshold=0.85)]
        
        if not suitable_nodes:
            logger.warning("❌ PowerOptimized: No suitable nodes")
            return None
        
        # Rank by power efficiency
        best_node = None
        best_efficiency = float('inf')
        
        for node in suitable_nodes:
            node_type = self.extract_node_type(node.name)
            efficiency = self.power_efficiency_ranking.get(node_type, 100.0)
            
            # Boost efficiency for nodes with relevant capabilities
            if 'inference' in deployment_name.lower() and node_type == 'coral':
                efficiency *= 0.6  # TPU is great for inference
            
            if efficiency < best_efficiency:
                best_efficiency = efficiency
                best_node = node
        
        if best_node:
            estimated_power = self.estimate_power_consumption(best_node, deployment_name)
            logger.info(f"🔋 PowerOptimized selected: {best_node.name} "
                       f"(efficiency: {best_efficiency:.1f}W, estimated power: {estimated_power:.1f}W)")
        
        return best_node
    
    def estimate_power_consumption(self, node: Any, deployment_name: str) -> float:
        """Estimate power consumption for workload on node"""
        node_type = self.extract_node_type(node.name)
        
        # Estimate moderate utilization
        estimated_cpu_util = 0.6
        estimated_gpu_util = 0.1 if 'gpu' in deployment_name.lower() else 0.0
        estimated_network_util = 0.2
        estimated_memory_util = 0.4
        
        return self.power_oracle.predict_power(
            node_type, estimated_cpu_util, estimated_gpu_util,
            estimated_network_util, estimated_memory_util
        )


class PerformanceOptimizedNodeSelector(NodeSelector):
    """Selects highest performance node"""
    
    def __init__(self, env, power_oracle):
        super().__init__(env, power_oracle)
        
        # Performance ranking (higher = better performance)
        self.performance_ranking = {
            'xeongpu': 10,    # Best performance
            'xeoncpu': 9,     
            'nuc': 8,         
            'nx': 7,          
            'tx2': 6,         
            'rockpi': 5,      
            'nano': 4,        
            'rpi4': 3,        
            'rpi3': 2,        
            'coral': 1        # Specialized only
        }
        
        # Expected execution time factors (lower = faster)
        self.execution_time_factors = {
            'xeongpu': 0.3,   
            'xeoncpu': 0.4,   
            'nuc': 0.5,       
            'nx': 0.6,        
            'tx2': 0.7,       
            'rockpi': 1.0,    # Baseline
            'nano': 1.2,      
            'rpi4': 1.8,      
            'rpi3': 2.5,      
            'coral': 0.8      # Good for inference only
        }
    
    def select_best_node(self, deployment_name: str, available_nodes: List[Any]) -> Optional[Any]:
        """Select highest performance available node"""
        
        # Filter nodes with sufficient resources (stricter for performance)
        suitable_nodes = [node for node in available_nodes 
                         if self.has_sufficient_resources(node, deployment_name, 
                                                        cpu_threshold=0.7, 
                                                        memory_threshold=0.7)]
        
        if not suitable_nodes:
            logger.warning("❌ PerformanceOptimized: No suitable nodes")
            return None
        
        # Rank by performance
        best_node = None
        best_performance = 0
        best_exec_time = float('inf')
        
        for node in suitable_nodes:
            node_type = self.extract_node_type(node.name)
            base_performance = self.performance_ranking.get(node_type, 1)
            
            # Adjust performance score based on workload
            adjusted_performance = base_performance
            
            # Boost for GPU workloads
            if 'inference' in deployment_name.lower() and node_type in ['xeongpu', 'nx', 'tx2', 'nano']:
                adjusted_performance += 2
            
            # Boost for training workloads on high-CPU nodes
            if 'training' in deployment_name.lower() and node_type in ['xeoncpu', 'xeongpu', 'nuc']:
                adjusted_performance += 3
            
            # Calculate expected execution time
            exec_time = self.estimate_execution_time(node, deployment_name)
            
            # Select based on performance score and execution time
            if adjusted_performance > best_performance or \
               (adjusted_performance == best_performance and exec_time < best_exec_time):
                best_performance = adjusted_performance
                best_exec_time = exec_time
                best_node = node
        
        if best_node:
            estimated_power = self.estimate_power_consumption(best_node, deployment_name)
            logger.info(f"⚡ PerformanceOptimized selected: {best_node.name} "
                       f"(performance: {best_performance}, exec time: {best_exec_time:.1f}ms, "
                       f"power cost: {estimated_power:.1f}W)")
        
        return best_node
    
    def estimate_execution_time(self, node: Any, deployment_name: str) -> float:
        """Estimate execution time on node"""
        node_type = self.extract_node_type(node.name)
        base_time = 500.0  # Base execution time in ms
        
        time_factor = self.execution_time_factors.get(node_type, 1.0)
        
        # Adjust for workload type
        if 'inference' in deployment_name.lower():
            if node_type == 'coral':  # TPU optimized
                time_factor *= 0.6
            elif node_type in ['xeongpu', 'nx', 'tx2']:  # GPU acceleration
                time_factor *= 0.7
        
        if 'training' in deployment_name.lower():
            if node_type in ['xeongpu', 'xeoncpu']:  # High compute
                time_factor *= 0.5
            elif node_type in ['rpi3', 'rpi4', 'coral']:  # Poor for training
                time_factor *= 3.0
        
        return base_time * time_factor
    
    def estimate_power_consumption(self, node: Any, deployment_name: str) -> float:
        """Estimate power consumption (for cost awareness)"""
        node_type = self.extract_node_type(node.name)
        
        # High utilization for performance optimization
        estimated_cpu_util = 0.8
        estimated_gpu_util = 0.6 if 'gpu' in deployment_name.lower() else 0.0
        estimated_network_util = 0.3
        estimated_memory_util = 0.7
        
        return self.power_oracle.predict_power(
            node_type, estimated_cpu_util, estimated_gpu_util,
            estimated_network_util, estimated_memory_util
        )


class HybridNodeSelector(NodeSelector):
    """Combines multiple selection criteria with weights"""
    
    def __init__(self, env, power_oracle, power_weight: float = 0.3, 
                 performance_weight: float = 0.7):
        super().__init__(env, power_oracle)
        self.power_weight = power_weight
        self.performance_weight = performance_weight
        
        # Initialize sub-selectors
        self.power_selector = PowerOptimizedNodeSelector(env, power_oracle)
        self.performance_selector = PerformanceOptimizedNodeSelector(env, power_oracle)
    
    def select_best_node(self, deployment_name: str, available_nodes: List[Any]) -> Optional[Any]:
        """Select node using hybrid scoring"""
        
        suitable_nodes = [node for node in available_nodes 
                         if self.has_sufficient_resources(node, deployment_name)]
        
        if not suitable_nodes:
            return None
        
        best_node = None
        best_score = -1
        
        for node in suitable_nodes:
            # Calculate power score (lower power = higher score)
            power_consumption = self.power_selector.estimate_power_consumption(node, deployment_name)
            power_score = 1.0 / max(power_consumption, 0.1)  # Invert so lower power = higher score
            
            # Calculate performance score
            exec_time = self.performance_selector.estimate_execution_time(node, deployment_name)
            performance_score = 1.0 / max(exec_time, 1.0)  # Invert so lower time = higher score
            
            # Combine scores
            hybrid_score = (self.power_weight * power_score + 
                          self.performance_weight * performance_score)
            
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_node = node
        
        if best_node:
            logger.info(f"⚖️  Hybrid selected: {best_node.name} (score: {best_score:.3f})")
        
        return best_node


def create_node_selector(strategy_name: str, env, power_oracle) -> NodeSelector:
    """Factory function to create appropriate node selector"""
    
    selectors = {
        'basic': FirstFitNodeSelector,
        'power': PowerOptimizedNodeSelector,
        'performance': PerformanceOptimizedNodeSelector,
        'hybrid': HybridNodeSelector
    }
    
    if strategy_name not in selectors:
        logger.warning(f"Unknown selector: {strategy_name}, using basic")
        strategy_name = 'basic'
    
    return selectors[strategy_name](env, power_oracle)