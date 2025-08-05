import logging
from typing import Optional, Any, List, Tuple
from ..base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization

logger = logging.getLogger(__name__)

class StandardFirstFitBinPacker(BaseAutoscaler):
    """
    Improved Standard First-Fit Bin Packing Strategy
    
    Implementation based on:
    - "Best-Fit Decreasing" algorithm from Coffman et al. (2013)
    - "Energy-Efficient Resource Allocation" by Beloglazov & Buyya (2012)
    - "Container Scheduling" approaches from Kubernetes research
    
    Strategy: Fill Partially Used Nodes First
    1. Sort nodes by current utilization (ascending) - partially used first
    2. Apply resource requirements checking 
    3. Select first node that fits (classic First-Fit)
    4. Promotes node consolidation and energy efficiency
    
    Benefits:
    - Better resource utilization through consolidation
    - Reduced energy consumption (fewer active nodes)
    - Improved cache locality and network efficiency
    - Industry-standard approach used in Kubernetes
    """

    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, "StandardFirstFitBinPacker")

        # Bin packing thresholds
        self.scale_up_threshold = 15
        self.scale_down_threshold = 3
        self.response_time_threshold = 800
        
        # Consolidation parameters (from research)
        self.min_utilization_threshold = 0.1    # Consider nodes with >10% usage as "partially used"
        self.max_utilization_threshold = 0.8    # Don't overload nodes beyond 80%
        self.consolidation_preference = True     # Prefer filling existing nodes
        
        logger.info(f"🗂️  Initialized First-Fit with consolidation (fill partially used first)")
    
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                             current_load: float, avg_response_time: float) -> str:
        """Standard scaling decision logic"""
        
        if current_load > self.scale_up_threshold or avg_response_time > self.response_time_threshold:
            if current_replicas < self.max_replicas:
                logger.info(f"📈 SCALING UP {deployment_name}: load={current_load:.1f} RPS, "
                           f"response_time={avg_response_time:.1f}ms")
                return "scale_up"
        
        elif current_load < self.scale_down_threshold and avg_response_time < 200:
            if current_replicas > self.min_replicas:
                logger.info(f"📉 SCALING DOWN {deployment_name}: load={current_load:.1f} RPS")
                return "scale_down"
        
        return "no_action"
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """
        Fill Partially Used Nodes First Algorithm
        
        Based on "Best-Fit" bin packing with consolidation optimization:
        1. Get all available compute nodes
        2. Sort by current utilization (ascending) - partially used first
        3. Apply First-Fit: select first node that meets resource requirements
        """
        
        try:
            # Get all available compute nodes
            available_nodes = self.get_available_nodes()
            
            if not available_nodes:
                logger.error("❌ No available nodes found!")
                return None
            
            # Apply bin packing algorithm: sort nodes for optimal fitting
            sorted_nodes = self.sort_nodes_for_bin_packing(available_nodes)
            
            # Debug: Log the sorting strategy
            logger.debug(f"🗂️  Node utilization ordering (fill partially used first):")
            for i, (node, util_score, util_data) in enumerate(sorted_nodes[:5]):
                logger.debug(f"  {i}: {node.name} - CPU: {util_data['cpu']:.1%}, "
                           f"Score: {util_score:.3f}")
            
            # First-Fit: iterate through sorted nodes and pick first suitable one
            for node, util_score, util_data in sorted_nodes:
                if self.has_sufficient_resources(node, deployment_name):
                    node_type = self.extract_node_type(node.name)
                    
                    # Log selection rationale
                    if util_data['cpu'] > self.min_utilization_threshold:
                        selection_reason = f"CONSOLIDATION: Filling partially used node"
                    else:
                        selection_reason = f"NEW NODE: No suitable partially used nodes"
                    
                    logger.info(f"✅ First-Fit Selected: {node.name} ({node_type}) - "
                               f"CPU: {util_data['cpu']:.1%}, Memory: {util_data['memory']:.1%}")
                    logger.info(f"   📋 Reason: {selection_reason}")
                    
                    return node
            
            logger.warning("❌ First-Fit: No nodes have sufficient resources")
            return None
            
        except Exception as e:
            logger.error(f"❌ Node selection error: {e}")
            return None
    
    def sort_nodes_for_bin_packing(self, nodes: List[Any]) -> List[Tuple[Any, float, dict]]:
        """
        Sort nodes for optimal bin packing (fill partially used first)
        
        Algorithm based on Beloglazov & Buyya (2012) energy-efficient allocation:
        1. Partially used nodes first (ascending utilization)
        2. Then unused nodes (by capacity - highest first)
        3. Skip heavily loaded nodes
        
        Returns: List of (node, sort_score, utilization_data) tuples
        """
        
        node_scores = []
        
        for node in nodes:
            try:
                util_data = self.get_node_utilization(node)
                cpu_util = util_data.get('cpu', 0.0)
                memory_util = util_data.get('memory', 0.0)
                
                # Skip nodes that are too heavily loaded
                if cpu_util > self.max_utilization_threshold or memory_util > self.max_utilization_threshold:
                    continue
                
                # Calculate bin packing score for sorting
                if cpu_util >= self.min_utilization_threshold:
                    # Partially used nodes: sort by utilization (ascending)
                    # Lower utilization = higher priority for filling
                    score = cpu_util + (memory_util * 0.5)  # CPU weighted more heavily
                    score_category = "partially_used"
                else:
                    # Unused nodes: sort by capacity (descending - bigger bins first)
                    node_type = self.extract_node_type(node.name)
                    capacity_score = self.get_node_capacity_score(node_type)
                    score = 2.0 + (1.0 - capacity_score)  # Offset to put after partially used
                    score_category = "unused"
                
                node_scores.append((node, score, util_data, score_category))
                
            except Exception as e:
                logger.debug(f"❌ Error evaluating node {node.name}: {e}")
                continue
        
        # Sort by score (ascending) - partially used with low utilization first
        sorted_nodes = sorted(node_scores, key=lambda x: x[1])
        
        # Debug logging
        partially_used = sum(1 for _, _, _, cat in sorted_nodes if cat == "partially_used")
        unused = len(sorted_nodes) - partially_used
        logger.debug(f"🗂️  Bin packing: {partially_used} partially used, {unused} unused nodes")
        
        # Return without category for compatibility
        return [(node, score, util_data) for node, score, util_data, _ in sorted_nodes]
    
    def get_node_capacity_score(self, node_type: str) -> float:
        """
        Get normalized capacity score for node type (0.0 to 1.0)
        Used for sorting unused nodes by capacity
        """
        capacity_ranking = {
            'xeongpu': 1.0,    # Highest capacity
            'xeoncpu': 0.9,
            'nuc': 0.8,
            'nx': 0.7,
            'tx2': 0.6,
            'rockpi': 0.5,
            'nano': 0.4,
            'coral': 0.3,
            'rpi4': 0.2,
            'rpi3': 0.1        # Lowest capacity
        }
        return capacity_ranking.get(node_type, 0.5)
    
    def has_sufficient_resources(self, node, deployment_name: str) -> bool:
        """
        Enhanced resource checking with function-specific requirements
        
        Based on Kubernetes resource allocation practices:
        - Different thresholds for different workload types
        - Function compatibility checking
        - Headroom reservation for system processes
        """
        try:
            current_utilization = self.get_node_utilization(node)
            node_type = self.extract_node_type(node.name)
            
            # Base resource thresholds (conservative for stability)
            cpu_threshold = self.max_utilization_threshold
            memory_threshold = self.max_utilization_threshold
            
            # Adjust thresholds based on deployment requirements
            if 'inference' in deployment_name.lower():
                # Inference tasks are usually lighter and more predictable
                cpu_threshold = 0.85
                memory_threshold = 0.85
            elif 'training' in deployment_name.lower():
                # Training tasks need more resources and are bursty
                cpu_threshold = 0.7
                memory_threshold = 0.7
                # Block training on low-end devices
                if node_type in ['rpi3', 'rpi4', 'coral']:
                    logger.debug(f"❌ {node_type} incompatible with training workload")
                    return False
            
            # Check actual availability
            cpu_available = current_utilization.get('cpu', 0) < cpu_threshold
            memory_available = current_utilization.get('memory', 0) < memory_threshold
            
            # Function compatibility check
            is_compatible = self.check_function_compatibility(node_type, deployment_name)
            
            result = cpu_available and memory_available and is_compatible
            
            if not result:
                logger.debug(f"❌ {node.name} insufficient: "
                           f"CPU: {current_utilization.get('cpu', 0):.1%} < {cpu_threshold:.1%}? {cpu_available} "
                           f"Mem: {current_utilization.get('memory', 0):.1%} < {memory_threshold:.1%}? {memory_available} "
                           f"Compatible: {is_compatible}")
            
            return result
            
        except Exception as e:
            logger.debug(f"❌ Resource check failed for {node.name}: {e}")
            return False  # Conservative: reject if can't determine
    
    def check_function_compatibility(self, node_type: str, deployment_name: str) -> bool:
        """
        Check function-node compatibility based on hardware capabilities
        """
        
        # GPU-required functions
        if any(keyword in deployment_name.lower() 
               for keyword in ['gpu', '-training']):  # Most training needs GPU
            gpu_capable = node_type in ['xeongpu', 'nx', 'tx2', 'nano']
            if not gpu_capable:
                return False
        
        # TPU-optimized functions
        if 'inference' in deployment_name.lower() and 'tflite' not in deployment_name.lower():
            # Coral TPU is excellent for inference but only for supported models
            if node_type == 'coral':
                # For now, assume coral is good for most inference
                return True
        
        # Memory-intensive functions (avoid very low-memory devices)
        if any(keyword in deployment_name.lower() 
               for keyword in ['resnet', 'training']):
            if node_type in ['rpi3']:  # Very limited memory
                return False
        
        return True
    
    def extract_node_type(self, node_name: str) -> str:
        """Extract node type from node name"""
        node_types = ['xeongpu', 'xeoncpu', 'nuc', 'nx', 'tx2', 'rockpi', 
                     'nano', 'coral', 'rpi4', 'rpi3']
        
        node_name_lower = node_name.lower()
        for node_type in node_types:
            if node_type in node_name_lower:
                return node_type
        return 'unknown'
    
    def get_available_nodes(self):
        """Get all available compute nodes"""
        try:
            if hasattr(self.env, 'topology') and hasattr(self.env.topology, 'nodes'):
                all_nodes = list(self.env.topology.nodes)
            else:
                logger.error("❌ Cannot access topology nodes")
                return []
            
            # Filter to compute nodes only
            compute_nodes = [node for node in all_nodes if self.is_compute_node(node)]
            
            logger.debug(f"🔍 Found {len(compute_nodes)} compute nodes out of {len(all_nodes)} total")
            return compute_nodes
            
        except Exception as e:
            logger.error(f"❌ Error getting available nodes: {e}")
            return []
    
    def is_compute_node(self, node) -> bool:
        """Check if node is a compute node (not registry/infrastructure)"""
        return (hasattr(node, 'capacity') and 
                not any(keyword in node.name.lower() 
                       for keyword in ['registry', 'link', 'switch', 'shared']))
    
    def get_node_utilization(self, node) -> dict:
        """Get current node resource utilization from ResourceState"""
        try:
            real_utilization = get_current_utilization(self.env, node.name)
            
            if real_utilization is None:
                logger.debug(f"❌ No utilization data for {node.name}, assuming idle")
                return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}
            
            return real_utilization
            
        except Exception as e:
            logger.debug(f"❌ Exception getting utilization for {node.name}: {e}")
            return {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0, 'network': 0.0}