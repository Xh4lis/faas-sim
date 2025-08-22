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

        # Power-aware thresholds (balanced for efficiency)
        self.scale_up_threshold = 8     # Scale up at 8 RPS (more responsive)
        self.scale_down_threshold = 3   # Scale down below 3 RPS
        self.response_time_threshold = 800  # Accept 800ms response time (much better than 1200ms)

        # Consolidation parameters
        self.min_utilization_for_new_node = 0.60  # Only use new nodes if existing >60% utilized
        self.max_utilization_per_node = 0.85      # Don't overload beyond 85%
        self.consolidation_bonus = 2.0            # Prefer existing nodes with replicas

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
        """Power-aware scaling decision - prioritize energy efficiency"""
        
        # Calculate current power efficiency
        current_power_efficiency = self.calculate_power_efficiency(deployment_name)
        
        logger.debug(f"🔋 PowerOptimized: {deployment_name} - Load: {current_load:.1f} RPS, "
                    f"Response: {avg_response_time:.1f}ms, Power Efficiency: {current_power_efficiency:.2f} W/RPS")
        
        # Scale up conditions (responsive but power-aware)
        if (current_load > self.scale_up_threshold or 
            avg_response_time > self.response_time_threshold) and \
           current_replicas < self.max_replicas:
            return "scale_up"
        
        # Scale down conditions (conservative to maintain power efficiency)
        elif current_load < self.scale_down_threshold and \
             avg_response_time < (self.response_time_threshold * 0.6) and \
             current_replicas > self.min_replicas:
            return "scale_down"
        
        return "no_action"
    
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """Enhanced node selection with consolidation preference"""
        available_nodes = self.get_available_nodes()
        
        # STEP 1: Try to consolidate on existing nodes first
        existing_replica_nodes = self.get_nodes_with_replicas(deployment_name)
        for node in existing_replica_nodes:
            if self.can_add_replica_safely(node, deployment_name):
                logger.info(f"🔋 CONSOLIDATION: Adding replica to existing node {node.name}")
                return node
        
        # STEP 2: Try nodes already running ANY replicas (bin packing)
        busy_nodes = self.get_nodes_with_any_replicas()
        power_ranked_busy = self.rank_nodes_by_power_efficiency(busy_nodes)
        
        for node, power_rank in power_ranked_busy:
            if self.can_add_replica_safely(node, deployment_name):
                logger.info(f"🔋 BIN-PACKING: Adding to busy low-power node {node.name}")
                return node
        
        # STEP 3: Only then consider new nodes (with power efficiency)
        empty_nodes = [n for n in available_nodes if not self.has_any_replicas(n)]
        power_ranked_empty = self.rank_nodes_by_power_efficiency(empty_nodes)
        
        for node, power_rank in power_ranked_empty:
            if self.has_sufficient_resources(node, deployment_name):
                logger.info(f"🔋 NEW-NODE: Using new low-power node {node.name}")
                return node
        
        return None
    
    def can_add_replica_safely(self, node, deployment_name: str) -> bool:
        """Check if we can add replica without overloading"""
        utilization = self.get_node_utilization(node)
        
        # More conservative thresholds for consolidation
        cpu_safe = utilization.get('cpu', 0) < self.max_utilization_per_node
        memory_safe = utilization.get('memory', 0) < self.max_utilization_per_node
        
        return cpu_safe and memory_safe
    
    def get_nodes_with_replicas(self, deployment_name: str) -> List[Any]:
        """Get nodes that already have replicas of this deployment"""
        nodes_with_replicas = []
        replicas = self.faas.get_replicas(deployment_name)
        
        for replica in replicas:
            if replica.node not in nodes_with_replicas:
                nodes_with_replicas.append(replica.node)
        
        return nodes_with_replicas

    def has_sufficient_resources(self, node, deployment_name: str) -> bool:
        """Check if node has sufficient CPU/memory resources - more lenient for spreading"""
        try:
            current_utilization = self.get_node_utilization(node)
            
            # More lenient thresholds to enable better spreading
            cpu_available = current_utilization.get('cpu', 0) < 0.85  # 85% instead of 90%
            memory_available = current_utilization.get('memory', 0) < 0.80  # 80% instead of 85%
            
            logger.debug(f"🔍 Resource check for {node.name}: CPU={current_utilization.get('cpu', 0):.2f} (<0.85?), Memory={current_utilization.get('memory', 0):.2f} (<0.80?)")
            
            return cpu_available and memory_available
        except Exception as e:
            logger.warning(f"⚠️ Resource check failed for {node.name}: {e}, assuming available")
            return True  # Be optimistic to allow spreading
    
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
    
    def rank_nodes_by_workload_efficiency(self, nodes: List[Any], deployment_name: str) -> List[Tuple[Any, float]]:
        """Rank nodes by combined power efficiency and workload suitability"""
        ranked_nodes = []
        
        for node in nodes:
            node_type = self.extract_node_type(node.name)
            base_power_rank = self.power_efficiency_ranking.get(node_type, 99)
            
            # Workload-specific adjustments (lower score = better)
            workload_penalty = 0
            
            if 'inference' in deployment_name.lower():
                # Inference workloads benefit from acceleration
                if node_type in ['coral']:  # TPU acceleration
                    workload_penalty = -3  # Bonus for TPU
                elif node_type in ['nano', 'nx']:  # GPU acceleration  
                    workload_penalty = -2  # Bonus for GPU
                elif node_type in ['rpi3', 'rpi4']:  # CPU-only, slow
                    workload_penalty = +4  # Penalty for slow CPU
                    
            elif 'training' in deployment_name.lower():
                # Training needs high compute power
                if node_type in ['nx', 'nuc']:  # Good compute
                    workload_penalty = -2  # Bonus 
                elif node_type in ['rpi3', 'coral']:  # Bad for training
                    workload_penalty = +6  # Heavy penalty
                    
            elif 'fio' in deployment_name.lower():
                # I/O intensive workloads
                if node_type in ['nuc', 'rockpi']:  # Good I/O
                    workload_penalty = -1  # Small bonus
                elif node_type in ['rpi3']:  # Poor I/O
                    workload_penalty = +3  # Penalty
            
            # Add current utilization penalty (spread load)
            try:
                utilization = self.get_node_utilization(node)
                util_penalty = (utilization.get('cpu', 0) + utilization.get('memory', 0)) * 2
            except:
                util_penalty = 0
            
            combined_score = base_power_rank + workload_penalty + util_penalty
            ranked_nodes.append((node, combined_score))
        
        # Sort by combined score (ascending - best first)
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
            return False  
    
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