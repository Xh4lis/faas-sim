import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class ScalingMetricsCollector:
    """
    Collects and analyzes metrics for scaling decisions
    Integrates with your existing power monitoring system
    """
    
    def __init__(self, env, window_size: int = 10):
        self.env = env
        self.window_size = window_size  # Number of measurements to keep for moving averages
        
        # Metrics storage with time windows
        self.load_history = defaultdict(lambda: deque(maxlen=window_size))
        self.response_time_history = defaultdict(lambda: deque(maxlen=window_size))
        self.power_history = defaultdict(lambda: deque(maxlen=window_size))
        self.utilization_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # Scaling event tracking
        self.scaling_events = []
        self.replica_history = defaultdict(lambda: deque(maxlen=window_size))
        
        logger.info(f"📊 ScalingMetricsCollector initialized with window size {window_size}")
    
    def record_load_metrics(self, deployment_name: str, current_rps: float, 
                           response_time: float):
        """Record load and performance metrics for a deployment"""
        timestamp = self.env.now
        
        self.load_history[deployment_name].append({
            'timestamp': timestamp,
            'rps': current_rps,
            'response_time': response_time
        })
        
        self.response_time_history[deployment_name].append(response_time)
        
        logger.debug(f"📈 Recorded load metrics for {deployment_name}: "
                    f"{current_rps:.1f} RPS, {response_time:.1f}ms")
    
    def record_power_metrics(self, node_name: str, power_watts: float, 
                           utilization: Dict[str, float]):
        """Record power and utilization metrics for a node"""
        timestamp = self.env.now
        
        self.power_history[node_name].append({
            'timestamp': timestamp,
            'power_watts': power_watts,
            'utilization': utilization.copy()
        })
        
        self.utilization_history[node_name].append(utilization.copy())
        
        logger.debug(f"⚡ Recorded power metrics for {node_name}: "
                    f"{power_watts:.1f}W, CPU: {utilization.get('cpu', 0):.1%}")
    
    def record_scaling_event(self, deployment_name: str, action: str, 
                           old_replicas: int, new_replicas: int, 
                           selected_node: str = None, reason: str = None):
        """Record a scaling event"""
        event = {
            'timestamp': self.env.now,
            'deployment': deployment_name,
            'action': action,
            'old_replicas': old_replicas,
            'new_replicas': new_replicas,
            'selected_node': selected_node,
            'reason': reason
        }
        
        self.scaling_events.append(event)
        self.replica_history[deployment_name].append(new_replicas)
        
        logger.info(f"📝 Scaling event: {action} {deployment_name} "
                   f"({old_replicas} → {new_replicas}) on {selected_node}")
    
    def get_average_load(self, deployment_name: str) -> float:
        """Get average RPS load over the window"""
        if deployment_name not in self.load_history or not self.load_history[deployment_name]:
            return 0.0
        
        loads = [entry['rps'] for entry in self.load_history[deployment_name]]
        return np.mean(loads)
    
    def get_average_response_time(self, deployment_name: str) -> float:
        """Get average response time over the window"""
        if deployment_name not in self.response_time_history or not self.response_time_history[deployment_name]:
            return 200.0  # Default reasonable response time
        
        return np.mean(list(self.response_time_history[deployment_name]))
    
    def get_load_trend(self, deployment_name: str) -> str:
        """Analyze load trend (increasing, decreasing, stable)"""
        if deployment_name not in self.load_history or len(self.load_history[deployment_name]) < 3:
            return "stable"
        
        recent_loads = [entry['rps'] for entry in list(self.load_history[deployment_name])[-3:]]
        
        if len(recent_loads) < 2:
            return "stable"
        
        # Calculate trend
        trend = np.mean(np.diff(recent_loads))
        
        if trend > 2.0:  # Increasing by more than 2 RPS
            return "increasing"
        elif trend < -2.0:  # Decreasing by more than 2 RPS
            return "decreasing"
        else:
            return "stable"
    
    def get_node_power_efficiency(self, node_name: str) -> Dict[str, float]:
        """Calculate power efficiency metrics for a node"""
        if node_name not in self.power_history or not self.power_history[node_name]:
            return {'efficiency': 0.0, 'avg_power': 0.0, 'avg_utilization': 0.0}
        
        recent_power = list(self.power_history[node_name])
        
        avg_power = np.mean([entry['power_watts'] for entry in recent_power])
        avg_cpu_util = np.mean([entry['utilization'].get('cpu', 0) for entry in recent_power])
        
        # Efficiency = useful work / power consumption
        # Higher CPU utilization with lower power = more efficient
        efficiency = avg_cpu_util / max(avg_power, 0.1)  # Avoid division by zero
        
        return {
            'efficiency': efficiency,
            'avg_power': avg_power,
            'avg_utilization': avg_cpu_util
        }
    
    def get_deployment_power_consumption(self, deployment_name: str) -> float:
        """Get total power consumption for all replicas of a deployment"""
        total_power = 0.0
        
        try:
            # Get all replicas for this deployment
            replicas = self.env.faas.get_replicas(deployment_name)
            
            for replica in replicas:
                node_name = replica.node.name
                node_efficiency = self.get_node_power_efficiency(node_name)
                total_power += node_efficiency['avg_power']
        
        except Exception as e:
            logger.debug(f"Could not calculate power for {deployment_name}: {e}")
        
        return total_power
    
    def should_scale_up(self, deployment_name: str, strategy_thresholds: Dict[str, float]) -> tuple:
        """Determine if deployment should scale up based on collected metrics"""
        # Get current metrics
        avg_load = self.get_average_load(deployment_name)
        avg_response_time = self.get_average_response_time(deployment_name)
        load_trend = self.get_load_trend(deployment_name)
        current_power = self.get_deployment_power_consumption(deployment_name)
        
        # Check against thresholds
        reasons = []
        should_scale = False
        
        if avg_load > strategy_thresholds.get('rps_threshold', 20):
            reasons.append(f"High load: {avg_load:.1f} > {strategy_thresholds['rps_threshold']}")
            should_scale = True
        
        if avg_response_time > strategy_thresholds.get('response_time_threshold', 1000):
            reasons.append(f"High response time: {avg_response_time:.1f}ms > {strategy_thresholds['response_time_threshold']}ms")
            should_scale = True
        
        if load_trend == "increasing":
            reasons.append("Load trend is increasing")
            should_scale = True
        
        return should_scale, "; ".join(reasons) if reasons else "No scaling needed"
    
    def should_scale_down(self, deployment_name: str, strategy_thresholds: Dict[str, float]) -> tuple:
        """Determine if deployment should scale down"""
        avg_load = self.get_average_load(deployment_name)
        avg_response_time = self.get_average_response_time(deployment_name)
        load_trend = self.get_load_trend(deployment_name)
        
        reasons = []
        should_scale = False
        
        if avg_load < strategy_thresholds.get('scale_down_threshold', 5):
            reasons.append(f"Low load: {avg_load:.1f} < {strategy_thresholds['scale_down_threshold']}")
            should_scale = True
        
        if avg_response_time < strategy_thresholds.get('response_time_threshold', 1000) * 0.3:
            reasons.append(f"Low response time: {avg_response_time:.1f}ms")
            should_scale = True
        
        if load_trend == "decreasing":
            reasons.append("Load trend is decreasing")
            should_scale = True
        
        return should_scale, "; ".join(reasons) if reasons else "No scaling down needed"
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of scaling activity"""
        total_scale_ups = len([e for e in self.scaling_events if e['action'] == 'scale_up'])
        total_scale_downs = len([e for e in self.scaling_events if e['action'] == 'scale_down'])
        
        # Node selection frequency
        node_selection = defaultdict(int)
        for event in self.scaling_events:
            if event['selected_node']:
                node_selection[event['selected_node']] += 1
        
        return {
            'total_scaling_events': len(self.scaling_events),
            'scale_ups': total_scale_ups,
            'scale_downs': total_scale_downs,
            'most_selected_nodes': dict(sorted(node_selection.items(), 
                                             key=lambda x: x[1], reverse=True)[:5]),
            'scaling_activity_timeline': self.scaling_events[-10:]  # Last 10 events
        }
    
    def export_metrics_to_csv(self, output_dir: str = "."):
        """Export collected metrics to CSV files"""
        import pandas as pd
        import os
        
        # Export scaling events
        if self.scaling_events:
            scaling_df = pd.DataFrame(self.scaling_events)
            scaling_df.to_csv(os.path.join(output_dir, "scaling_events.csv"), index=False)
            logger.info(f"📄 Exported {len(self.scaling_events)} scaling events to CSV")
        
        # Export load history
        load_data = []
        for deployment, history in self.load_history.items():
            for entry in history:
                load_data.append({
                    'deployment': deployment,
                    'timestamp': entry['timestamp'],
                    'rps': entry['rps'],
                    'response_time': entry['response_time']
                })
        
        if load_data:
            load_df = pd.DataFrame(load_data)
            load_df.to_csv(os.path.join(output_dir, "load_metrics.csv"), index=False)
            logger.info(f"📄 Exported {len(load_data)} load measurements to CSV")
        
        # Export power efficiency data
        power_data = []
        for node, history in self.power_history.items():
            for entry in history:
                power_data.append({
                    'node': node,
                    'timestamp': entry['timestamp'],
                    'power_watts': entry['power_watts'],
                    'cpu_util': entry['utilization'].get('cpu', 0),
                    'memory_util': entry['utilization'].get('memory', 0),
                    'gpu_util': entry['utilization'].get('gpu', 0)
                })
        
        if power_data:
            power_df = pd.DataFrame(power_data)
            power_df.to_csv(os.path.join(output_dir, "scaling_power_metrics.csv"), index=False)
            logger.info(f"📄 Exported {len(power_data)} power measurements to CSV")