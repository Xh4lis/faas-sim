import logging
from typing import Optional, Any, List, Tuple
from ..base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization

logger = logging.getLogger(__name__)

class HighPerformanceShortTimeBinPacker(BaseAutoscaler):
    """
    High-Performance Short-Time Energy Optimization Strategy
    
    Energy optimization through execution speed:
    - Prioritizes nodes with highest computational performance (Xeon, NX, NUC)
    - Accepts higher power consumption for shorter execution times
    - Optimizes total energy = high_power × short_time
    - Suitable for time-critical applications where speed reduces total energy
    """
    
    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, "HighPerformanceShortTimeBinPacker")

        # Performance-focused thresholds (aggressive)
        self.scale_up_threshold = 5     # Scale up at 5 RPS (higher threshold)
        self.scale_down_threshold = 2    # Scale down below 2 RPS
        self.response_time_threshold = 500  # Target 500ms response time (aggressive)
        self.max_response_time = 800     # Never exceed 800ms

        # Define performance rankings (higher number = better performance)
        self.performance_ranking = {
            'xeongpu': 10,  # Best performance (GPU + high CPU)
            'xeoncpu': 9,   # Second best (high CPU)
            'nuc': 8,       # Good performance
            'nx': 7,        # Nvidia Jetson NX (good GPU)
            'tx2': 6,       # Nvidia Jetson TX2 (moderate GPU)
            'rockpi': 5,    # Moderate performance
            'nano': 4,      # Lower performance but has GPU
            'rpi4': 3,      # Limited performance
            'rpi3': 2,      # Older, limited performance
            'coral': 1      # Specialized for inference only
        }
        
        # Expected execution time factors (lower = faster)
        self.execution_time_factors = {
            'xeongpu': 0.3,   # 30% of baseline time
            'xeoncpu': 0.4,   # 40% of baseline time
            'nuc': 0.5,       # 50% of baseline time
            'nx': 0.6,        # 60% of baseline time
            'tx2': 0.7,       # 70% of baseline time
            'rockpi': 1.0,    # Baseline time
            'nano': 1.2,      # 120% of baseline time
            'rpi4': 1.8,      # 180% of baseline time
            'rpi3': 2.5,      # 250% of baseline time
            'coral': 0.8      # 80% for inference tasks only
        }
    
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                            current_load: float, avg_response_time: float) -> str:
        """Performance-optimized scaling decision - prioritize speed"""
        
        # Calculate current performance metrics
        avg_execution_time = self.calculate_average_execution_time(deployment_name)
        throughput = self.calculate_throughput(deployment_name)
        
        logger.debug(f"⚡ PerformanceOptimized: {deployment_name} - Load: {current_load:.1f} RPS, "
                    f"Response: {avg_response_time:.1f}ms, Exec Time: {avg_execution_time:.1f}ms, "
                    f"Throughput: {throughput:.1f} req/s")
        
        if (current_load > self.scale_up_threshold or
            avg_response_time > self.response_time_threshold) and \
           current_replicas < self.max_replicas:
            return "scale_up"
        
        # Conservative scale down conditions (keep performance nodes)
        elif current_load < self.scale_down_threshold and \
             avg_response_time < (self.response_time_threshold * 0.6) and \
             current_replicas > self.min_replicas:
            return "scale_down"
        
        return "no_action"
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """Performance-optimized: Select highest-performance available node - AVOID LOW-POWER DEVICES"""
        
        available_nodes = self.get_available_nodes()
        
        # Sort nodes by performance (highest performance first)
        performance_ranked_nodes = self.rank_nodes_by_performance(available_nodes, deployment_name)
        
        # Track attempts for debugging
        high_power_attempts = []
        
        # First pass: Only try high-power devices (>5W)
        for node, performance_score in performance_ranked_nodes:
            node_type = self.extract_node_type(node.name)
            
            if node_type in ['xeongpu', 'xeoncpu', 'nuc', 'nx', 'tx2']:  # High-power devices only
                if self.has_sufficient_resources(node, deployment_name):
                    estimated_exec_time = self.estimate_execution_time(node, deployment_name)
                    estimated_power = self.estimate_node_power_consumption(node, deployment_name)
                    
                    logger.info(f"⚡ HPST: Selected HIGH-POWER {node.name} "
                               f"(type: {node_type}, performance rank: {performance_score}, "
                               f"estimated exec time: {estimated_exec_time:.1f}ms, "
                               f"power cost: {estimated_power:.1f}W)")
                    return node
                else:
                    utilization = self.get_node_utilization(node)
                    high_power_attempts.append(f"{node.name}({node_type}): CPU={utilization.get('cpu', 0):.2f}")
        
        # Log why high-power devices were rejected
        logger.warning(f"⚠️ HPST: All high-power devices busy for {deployment_name}. "
                      f"Attempts: {high_power_attempts[:3]}")
        
        # Emergency fallback: Still prefer mid-power over low-power
        emergency_order = ['nano', 'rockpi', 'coral', 'rpi4', 'rpi3']  # Least bad to worst
        for preferred_type in emergency_order:
            for node, performance_score in performance_ranked_nodes:
                node_type = self.extract_node_type(node.name)
                if node_type == preferred_type:
                    # Relaxed resource check for emergency
                    util = self.get_node_utilization(node)
                    if util.get('cpu', 0) < 0.9 and util.get('memory', 0) < 0.9:
                        logger.error(f"🚨 HPST EMERGENCY: Using low-power {node.name} ({node_type}) "
                                   f"- no high-power devices available!")
                        return node
        
        logger.error("❌ HPST: No nodes available at all - system overloaded!")
        return None
    
    def rank_nodes_by_performance(self, nodes: List[Any], deployment_name: str) -> List[Tuple[Any, int]]:
        """Rank nodes by performance (higher is better) - HPST strongly avoids low-power devices"""
        ranked_nodes = []
        
        for node in nodes:
            node_type = self.extract_node_type(node.name)
            base_performance = self.performance_ranking.get(node_type, 1)
            
            # STRONG BIAS AGAINST LOW-POWER DEVICES (this is the key fix)
            # HPST should heavily penalize rockpi and rpi3 to force high-power device usage
            if node_type in ['rpi3', 'rpi4']:
                base_performance -= 15  # Heavy penalty for RPi devices
            elif node_type in ['rockpi', 'coral']:
                base_performance -= 10  # Strong penalty for other low-power devices
            elif node_type in ['nano']:
                base_performance -= 5   # Moderate penalty for nano
            
            # STRONG BOOST FOR HIGH-POWER DEVICES
            if node_type in ['xeongpu', 'xeoncpu']:
                base_performance += 10  # Major boost for server-grade hardware
            elif node_type in ['nuc', 'nx']:
                base_performance += 5   # Good boost for mid-range high-power devices
            elif node_type in ['tx2']:
                base_performance += 2   # Small boost for moderate-power devices
            
            # Boost ranking for GPU-capable nodes if deployment needs GPU
            if 'gpu' in deployment_name.lower() or 'inference' in deployment_name.lower():
                if node_type in ['xeongpu', 'nx', 'tx2', 'nano']:
                    base_performance += 3  # Bonus for GPU capability
            
            # Boost ranking for training workloads on high-CPU nodes
            if 'training' in deployment_name.lower():
                if node_type in ['xeoncpu', 'xeongpu', 'nuc']:
                    base_performance += 5  # Bonus for high CPU
            
            ranked_nodes.append((node, base_performance))
        
        # Sort by performance rank (descending - best performance first)
        ranked_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_nodes
    
    def estimate_execution_time(self, node: Any, deployment_name: str) -> float:
        """Estimate execution time on specific node"""
        node_type = self.extract_node_type(node.name)
        base_time = 500.0  # Base execution time in milliseconds
        
        time_factor = self.execution_time_factors.get(node_type, 1.0)
        
        # Adjust for workload type
        if 'inference' in deployment_name.lower():
            if node_type == 'coral':  # TPU optimized for inference
                time_factor *= 0.6
            elif node_type in ['xeongpu', 'nx', 'tx2']:  # GPU acceleration
                time_factor *= 0.7
        
        if 'training' in deployment_name.lower():
            if node_type in ['xeongpu', 'xeoncpu']:  # High compute for training
                time_factor *= 0.5
            elif node_type in ['rpi3', 'rpi4', 'coral']:  # Poor for training
                time_factor *= 3.0
        
        return base_time * time_factor
    
    def estimate_node_power_consumption(self, node: Any, deployment_name: str) -> float:
        """Estimate power consumption (for cost awareness)"""
        node_type = self.extract_node_type(node.name)
        
        # High utilization for performance optimization
        estimated_cpu_util = 0.8   # 80% CPU utilization
        estimated_gpu_util = 0.6 if 'gpu' in deployment_name.lower() else 0.0
        estimated_network_util = 0.3  # 30% network utilization
        estimated_memory_util = 0.7   # 70% memory utilization
        
        return self.power_oracle.predict_power(
            node_type, estimated_cpu_util, estimated_gpu_util,
            estimated_network_util, estimated_memory_util
        )
    
    def calculate_average_execution_time(self, deployment_name: str) -> float:
        #FIXME: Implement actual metrics integration
        """Calculate average execution time for deployment"""
        try:
            # This would integrate with your metrics system
            # For now, return estimated value based on current node types
            replicas = self.faas.get_replicas(deployment_name)
            if not replicas:
                return 500.0
            
            total_exec_time = 0.0
            for replica in replicas:
                node_type = self.extract_node_type(replica.node.name)
                exec_time = self.estimate_execution_time(replica.node, deployment_name)
                total_exec_time += exec_time
            
            return total_exec_time / len(replicas)
        except:
            return 500.0  # Default execution time
    
    def calculate_throughput(self, deployment_name: str) -> float:
        """Calculate current throughput (requests per second)"""
        try:
            current_load = self.get_current_load(deployment_name)
            avg_exec_time = self.calculate_average_execution_time(deployment_name)
            
            # Theoretical throughput = replicas * (1000ms / avg_exec_time)
            replicas = len(self.faas.get_replicas(deployment_name))
            theoretical_throughput = replicas * (1000.0 / max(avg_exec_time, 100))
            
            return min(current_load, theoretical_throughput)
        except:
            return 0.0
    
    def extract_node_type(self, node_name: str) -> str:
        """Extract node type from node name"""
        for node_type in self.performance_ranking.keys():
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
        """Check if node has sufficient resources (strict for performance) - REJECT LOW-POWER DEVICES"""
        try:
            current_utilization = self.get_node_utilization(node)
            node_type = self.extract_node_type(node.name)
            
            # HPST STRATEGY: COMPLETELY AVOID LOW-POWER DEVICES UNLESS ABSOLUTELY NECESSARY
            # This forces the strategy to use higher power devices
            if node_type in ['rpi3', 'rpi4', 'rockpi', 'coral']:
                # Only use low-power devices if system is under extreme stress
                if current_utilization.get('cpu', 0) < 0.3:  # Only if very lightly loaded
                    logger.warning(f"🚨 HPST: Reluctantly considering low-power device {node.name}")
                    return False  # Still reject - force high-power device usage
                else:
                    return False  # Never use when busy
            
            # Strict resource thresholds for performance on acceptable devices
            cpu_available = current_utilization.get('cpu', 0) < 0.7   # 70% CPU threshold
            memory_available = current_utilization.get('memory', 0) < 0.7  # 70% memory threshold
            
            # Reserve high-performance nodes for performance-critical tasks
            if node_type in ['xeongpu', 'xeoncpu'] and current_utilization.get('cpu', 0) > 0.5:
                return False  # Reserve high-end nodes
            
            # Prefer high-power devices (>5W nominal power)
            if node_type in ['xeongpu', 'xeoncpu', 'nuc', 'nx', 'tx2']:
                logger.debug(f"✅ HPST: Accepting high-power device {node.name} ({node_type})")
                return cpu_available and memory_available
            
            # Reject everything else (forces high-power device usage)
            logger.debug(f"❌ HPST: Rejecting device {node.name} ({node_type}) - not high-power enough")
            return False
            
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