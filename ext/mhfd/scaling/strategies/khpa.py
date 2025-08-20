import logging
from typing import Optional, Any, List, Tuple
from ..base_autoscaler import BaseAutoscaler
from ext.mhfd.power import get_current_utilization

logger = logging.getLogger(__name__)

class KubernetesStyleFirstFitBinPacker(BaseAutoscaler):
    """
    Kubernetes-Style First-Fit Bin Packing Strategy (Baseline)
    
    Implementation based on:
    - "Kubernetes Scheduling: Taxonomy and Challenges" (ACM Computing Surveys, 2022)
    - "Performance Analysis of Kubernetes Default Scheduler" (ICDCS, 2023)
    - "Energy-Aware Container Scheduling in Kubernetes" (IEEE Cloud, 2022)
    
    Algorithm: Standard Kubernetes Default Scheduler + HPA
    1. Resource-based filtering (CPU/Memory requests)
    2. Least-Requested Priority (spread workload)
    3. Node scoring and ranking
    4. First-Fit selection from ranked nodes
    5. HPA-style reactive scaling (CPU/response time based)
    
    This represents the current industry standard for comparison.
    """

    def __init__(self, env, faas_system, power_oracle):
        super().__init__(env, faas_system, power_oracle, "KubernetesStyleFirstFit")

        # HPA-style thresholds (Kubernetes defaults)
        self.scale_up_threshold = 20      # 20 RPS 
        self.scale_down_threshold = 5     # 5 RPS
        self.response_time_threshold = 500  # 500ms (typical web service SLA)
        
        # Kubernetes scheduler parameters
        self.cpu_request_threshold = 0.8    # 80% CPU utilization
        self.memory_request_threshold = 0.8  # 80% memory utilization
        self.least_requested_weight = 1.0    # Standard Kubernetes weight
        
        # HPA stabilization windows (from Kubernetes HPA v2)
        self.scale_up_stabilization = 15   # 15 seconds
        self.scale_down_stabilization = 300 # 5 minutes (Kubernetes default)
        
        logger.info(f"🎯 Initialized Kubernetes-Style First-Fit (industry baseline)")
    
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                             current_load: float, avg_response_time: float) -> str:
        """HPA-style scaling decision (reactive, threshold-based)"""
        
        # HPA v2 style metrics evaluation
        cpu_utilization = self.get_deployment_cpu_utilization(deployment_name)
        
        logger.debug(f"🎯 K8s-Style: {deployment_name} - Load: {current_load:.1f} RPS, "
                    f"Response: {avg_response_time:.1f}ms, CPU: {cpu_utilization:.1%}")
        
        # Scale up conditions (OR logic - any condition triggers)
        if ((current_load > self.scale_up_threshold) or 
            (avg_response_time > self.response_time_threshold) or
            (cpu_utilization > 0.7)) and \
           current_replicas < self.max_replicas:
            
            logger.info(f"📈 K8s-Style SCALE UP {deployment_name}: "
                       f"Load={current_load:.1f}>{self.scale_up_threshold} OR "
                       f"ResponseTime={avg_response_time:.1f}>{self.response_time_threshold} OR "
                       f"CPU={cpu_utilization:.1%}>70%")
            return "scale_up"
        
        # Scale down conditions (AND logic - all conditions must be met)
        elif (current_load < self.scale_down_threshold and 
              avg_response_time < (self.response_time_threshold * 0.5) and
              cpu_utilization < 0.3) and \
             current_replicas > self.min_replicas:
            
            logger.info(f"📉 K8s-Style SCALE DOWN {deployment_name}: "
                       f"Load={current_load:.1f}<{self.scale_down_threshold} AND "
                       f"ResponseTime={avg_response_time:.1f}<{self.response_time_threshold*0.5} AND "
                       f"CPU={cpu_utilization:.1%}<30%")
            return "scale_down"
        
        return "no_action"
    
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """
        Kubernetes Default Scheduler Algorithm
        
        1. Filtering Phase: Remove unsuitable nodes
        2. Scoring Phase: Score remaining nodes  
        3. Selection Phase: Pick highest scoring node
        """
        
        try:
            # Phase 1: Filtering (Kubernetes Predicates)
            available_nodes = self.get_available_nodes()
            suitable_nodes = self.filter_nodes_kubernetes_style(available_nodes, deployment_name)
            
            if not suitable_nodes:
                logger.warning("❌ K8s-Style: No suitable nodes after filtering")
                return None
            
            # Phase 2: Scoring (Kubernetes Priority Functions)
            scored_nodes = self.score_nodes_kubernetes_style(suitable_nodes, deployment_name)
            
            # Phase 3: Selection (highest score wins)
            best_node, best_score, score_breakdown = scored_nodes[0]
            
            logger.info(f"✅ K8s-Style Selected: {best_node.name} "
                       f"(score: {best_score:.2f}, breakdown: {score_breakdown})")
            
            return best_node
            
        except Exception as e:
            logger.error(f"❌ K8s-Style node selection error: {e}")
            return None
    
    def filter_nodes_kubernetes_style(self, nodes: List[Any], deployment_name: str) -> List[Any]:
        """
        Kubernetes Filtering Phase (Predicates)
        Remove nodes that can't schedule the pod
        """
        suitable_nodes = []
        
        for node in nodes:
            # Check resource requirements
            if not self.check_resource_fit(node, deployment_name):
                continue
                
            # Check node conditions (like Kubernetes NodeReady)
            if not self.check_node_conditions(node):
                continue
                
            # Check taints/tolerations (simplified)
            if not self.check_node_affinity(node, deployment_name):
                continue
            
            suitable_nodes.append(node)
        
        logger.debug(f"🎯 K8s-Style: {len(suitable_nodes)}/{len(nodes)} nodes passed filtering")
        return suitable_nodes
    
    def score_nodes_kubernetes_style(self, nodes: List[Any], deployment_name: str) -> List[Tuple[Any, float, dict]]:
        """
        Enhanced Kubernetes Default Scheduler with Device Class Awareness
        (Similar to Kubernetes 1.26+ Dynamic Resource Allocation)
        """
        scored_nodes = []
        
        for node in nodes:
            scores = {}
            
            # 1. LeastRequestedPriority (main Kubernetes priority)
            utilization = self.get_node_utilization(node)
            cpu_score = (1.0 - utilization.get('cpu', 0)) * 100
            memory_score = (1.0 - utilization.get('memory', 0)) * 100
            scores['least_requested'] = (cpu_score + memory_score) / 2
            
            # 2. BalancedResourceAllocation 
            cpu_util = utilization.get('cpu', 0)
            memory_util = utilization.get('memory', 0)
            balance_score = 100 - (abs(cpu_util - memory_util) * 100)
            scores['balanced_resource'] = max(0, balance_score)
            
            # 3. Device Class Awareness (NEW - Enhanced Kubernetes)
            node_type = self.extract_node_type(node.name)
            device_class_score = self.get_device_class_score(node_type, deployment_name)
            scores['device_class'] = device_class_score
            
            # Enhanced Kubernetes weights (device class aware)
            total_score = (
                scores['least_requested'] * 6.0 +      # Still main priority (spread)
                scores['device_class'] * 4.0 +         # Device class matching
                scores['balanced_resource'] * 2.0      # Resource balance
            ) / 12.0
            
            scored_nodes.append((node, total_score, scores))
        
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Debug top nodes with device class info
        logger.debug(f"🎯 Enhanced K8s-Style top 5 nodes for {deployment_name}:")
        for i, (node, score, breakdown) in enumerate(scored_nodes[:5]):
            node_type = self.extract_node_type(node.name)
            required_class = self.get_required_device_class(deployment_name)
            logger.debug(f"  {i+1}. {node.name} ({node_type}) → {required_class}: {score:.1f} {breakdown}")
        
        return scored_nodes
    
    def get_device_class_score(self, node_type: str, deployment_name: str) -> float:
        """
        Enhanced Kubernetes with device class awareness
        (Similar to Kubernetes 1.26+ Dynamic Resource Allocation)
        """
        
        # Define device classes (like Kubernetes device plugins)
        device_classes = {
            'high-compute': ['nuc', 'xeoncpu', 'xeongpu'],
            'gpu-accelerated': ['xeongpu', 'nx', 'tx2', 'nano'],
            'edge-optimized': ['coral', 'nano', 'rockpi'],
            'low-power': ['nano', 'coral', 'rockpi', 'rpi4']
        }
        
        # Function requirements (could be set via labels/annotations)
        function_requirements = {
            'resnet50-inference': 'gpu-accelerated',
            'resnet50-training': 'gpu-accelerated', 
            'speech-inference': 'edge-optimized',
            'python-pi': 'low-power',
            'fio': 'high-compute'
        }
        
        required_class = function_requirements.get(deployment_name, 'high-compute')
        
        if node_type in device_classes.get(required_class, []):
            logger.debug(f"✅ {node_type} matches {required_class} for {deployment_name}")
            return 80.0  # Good match
        else:
            logger.debug(f"❌ {node_type} doesn't match {required_class} for {deployment_name}")
            return 30.0  # Acceptable but not optimal
    
    def get_required_device_class(self, deployment_name: str) -> str:
        """Helper to get required device class for logging"""
        function_requirements = {
            'resnet50-inference': 'gpu-accelerated',
            'resnet50-training': 'gpu-accelerated', 
            'speech-inference': 'edge-optimized',
            'python-pi': 'low-power',
            'fio': 'low-power'
        }
        return function_requirements.get(deployment_name, 'high-compute')
    
    
    
    def get_basic_compatibility_score(self, node_type: str, deployment_name: str) -> float:
        """Minimal compatibility checking (true to Kubernetes)"""
        
        # Only enforce hard hardware requirements
        if 'gpu' in deployment_name.lower():
            return 60.0 if node_type in ['xeongpu', 'nx', 'tx2', 'nano'] else 10.0
        
        # All other workloads treated equally
        return 50.0
    
    def check_resource_fit(self, node: Any, deployment_name: str) -> bool:
        """Check if node has sufficient resources (Kubernetes style)"""
        try:
            utilization = self.get_node_utilization(node)
            
            # Kubernetes-style resource requests checking
            cpu_available = utilization.get('cpu', 0) < self.cpu_request_threshold
            memory_available = utilization.get('memory', 0) < self.memory_request_threshold
            
            # Function-specific resource requirements
            if 'training' in deployment_name.lower():
                # Training needs more resources
                cpu_available = utilization.get('cpu', 0) < 0.6
                memory_available = utilization.get('memory', 0) < 0.6
            
            return cpu_available and memory_available
            
        except Exception as e:
            logger.debug(f"❌ Resource check failed for {node.name}: {e}")
            return False
    
    def check_node_conditions(self, node: Any) -> bool:
        """Check node health conditions (simplified)"""
        # In real Kubernetes: Ready, MemoryPressure, DiskPressure, etc.
        # Here we just check if node is responsive
        try:
            utilization = self.get_node_utilization(node)
            # Node is healthy if we can get utilization data
            return utilization is not None
        except:
            return False
    
    def check_node_affinity(self, node: Any, deployment_name: str) -> bool:
        """Check node affinity/anti-affinity (simplified)"""
        node_type = self.extract_node_type(node.name)
        
        # Block obviously incompatible combinations
        if 'training' in deployment_name.lower() and node_type in ['coral', 'rpi3']:
            return False  # Training shouldn't run on very limited devices
            
        if 'gpu' in deployment_name.lower() and node_type not in ['xeongpu', 'nx', 'tx2', 'nano']:
            return False  # GPU workloads need GPU nodes
            
        return True
    
    def get_node_affinity_score(self, node_type: str, deployment_name: str) -> float:
        """Calculate node affinity score based on workload suitability"""
        base_score = 50.0  # Neutral score
        
        # Boost score for well-matched node-workload pairs
        if 'inference' in deployment_name.lower():
            if node_type == 'coral':  # TPU excellent for inference
                return 90.0
            elif node_type in ['nx', 'nano', 'tx2']:  # GPU good for inference
                return 80.0
            elif node_type in ['nuc', 'rockpi']:  # Decent CPU for inference
                return 70.0
        
        if 'training' in deployment_name.lower():
            if node_type in ['xeongpu', 'xeoncpu']:  # High-end for training
                return 90.0
            elif node_type in ['nuc', 'nx']:  # Decent for training
                return 70.0
            elif node_type in ['rpi3', 'rpi4']:  # Poor for training
                return 20.0
        
        return base_score
    
    def get_deployment_cpu_utilization(self, deployment_name: str) -> float:
        """Get average CPU utilization across all replicas of deployment"""
        try:
            replicas = self.faas.get_replicas(deployment_name)
            if not replicas:
                return 0.0
            
            total_cpu = 0.0
            for replica in replicas:
                utilization = self.get_node_utilization(replica.node)
                total_cpu += utilization.get('cpu', 0.0)
            
            return total_cpu / len(replicas)
        except:
            return 0.0
    
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