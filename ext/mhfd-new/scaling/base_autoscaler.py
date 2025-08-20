import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from sim.core import Environment
import pandas as pd

logger = logging.getLogger(__name__)

class BaseAutoscaler(ABC):
    """Base class for all autoscaling strategies"""
    """ABC provides:

        Interface enforcement - All strategies must implement required methods
        Template method pattern - Shared infrastructure with custom algorithms
        Polymorphism - Same interface, different behaviors
        Error prevention - Can't create incomplete implementations
        Code reuse - Common functionality in base class
        Extensibility - Easy to add new strategies
  """
    
    def __init__(self, env: Environment, faas_system, power_oracle, strategy_name: str):
        self.env = env
        self.faas = faas_system
        self.power_oracle = power_oracle
        self.strategy_name = strategy_name
        self.scaling_interval = 5  # Check every 5 seconds
        self.scaling_history = {}
        # Scaling configuration
        self.min_replicas = 1
        self.max_replicas = 500
        self.scale_up_threshold = 50  # RPS threshold for scaling up
        self.scale_down_threshold = 5   # RPS threshold for scaling down
        self.response_time_threshold = 1000  # ms
        
        logger.info(f"Initialized {strategy_name} autoscaler")
    
    def run(self):
        """Main autoscaling loop - SimPy process"""
        logger.info(f"🚀 Starting {self.strategy_name} autoscaler")
        
        while True:
            yield self.env.timeout(self.scaling_interval)
            yield from self.evaluate_scaling_decisions()
    
    def evaluate_scaling_decisions(self):
        """Evaluate scaling decisions for all deployments"""
        logger.debug(f"📊 {self.strategy_name}: Evaluating scaling at t={self.env.now:.1f}s")
        
        # Get deployments from FaaS system
        deployments = self.faas.get_deployments()
        
        if not deployments:
            logger.debug(f"No deployments found for scaling evaluation")
            return
        
        for deployment in deployments:
            deployment_name = deployment.name
            
            # Get current metrics
            current_replicas = len(self.faas.get_replicas(deployment_name))
            current_load = self.get_current_load(deployment_name)
            avg_response_time = self.get_average_response_time(deployment_name)
            
            # Log evaluation attempt
            self.record_scaling_evaluation(deployment_name, current_replicas, 
                                         current_load, avg_response_time)
            
            # Make scaling decision using strategy-specific logic
            decision = self.make_scaling_decision(
                deployment_name, current_replicas, current_load, avg_response_time
            )
            
            if decision == "scale_up":
                yield from self.scale_up_deployment(deployment_name)
            elif decision == "scale_down":
                yield from self.scale_down_deployment(deployment_name)
            else:
                # Record "no action" decisions too
                self.record_scaling_decision(deployment_name, "no_action", 
                                           current_replicas, None, current_load, 
                                           avg_response_time, "Within thresholds")
        df = self.get_detailed_metrics_df()
    
    def record_scaling_evaluation(self, deployment_name: str, current_replicas: int,
                                current_load: float, avg_response_time: float):
        """Record scaling evaluation metrics"""
        eval_data = {
            'timestamp': self.env.now,
            'strategy': self.strategy_name,
            'deployment': deployment_name,
            'current_replicas': current_replicas,
            'current_load_rps': current_load,
            'avg_response_time_ms': avg_response_time,
            'evaluation_type': 'scaling_check'
        }
        
        # Log to environment metrics
        self.env.metrics.log('scaling_evaluations', eval_data)
    
    def record_scaling_decision(self, deployment_name: str, action: str, 
                              new_replica_count: int, selected_node: Any,
                              current_load: float = 0.0, avg_response_time: float = 0.0,
                              reason: str = ""):
        """Record scaling decision for DataFrame analysis"""
        decision_data = {
            'timestamp': self.env.now,
            'strategy': self.strategy_name,
            'deployment': deployment_name,
            'action': action,  # scale_up, scale_down, no_action
            'old_replica_count': new_replica_count - (1 if action == "scale_up" else -1 if action == "scale_down" else 0),
            'new_replica_count': new_replica_count,
            'selected_node': selected_node.name if selected_node else None,
            'selected_node_type': self.extract_node_type(selected_node.name) if selected_node else None,
            'current_load_rps': current_load,
            'avg_response_time_ms': avg_response_time,
            'reason': reason,
            'scaling_direction': 1 if action == "scale_up" else -1 if action == "scale_down" else 0
        }
        
        # Log to environment metrics for CSV export
        self.env.metrics.log('scaling_decisions', decision_data)
        
        logger.info(f"📝 {self.strategy_name}: {action} {deployment_name} → {new_replica_count} replicas"
                   f" (load: {current_load:.1f} RPS, response: {avg_response_time:.1f}ms)")
    
    def extract_node_type(self, node_name: str) -> str:
        """Extract node type from node name"""
        node_types = ['coral', 'nano', 'rpi3', 'rpi4', 'rockpi', 'tx2', 'nx', 
                     'nuc', 'xeoncpu', 'xeongpu', 'registry']
        
        for node_type in node_types:
            if node_type in node_name.lower():
                return node_type
        return 'unknown'
    
    @abstractmethod
    def make_scaling_decision(self, deployment_name: str, current_replicas: int, 
                            current_load: float, avg_response_time: float) -> str:
        """Strategy-specific scaling decision logic"""
        pass
    
    @abstractmethod
    def select_node_for_scaling(self, deployment_name: str) -> Optional[Any]:
        """Strategy-specific node selection for scaling up"""
        pass
    
    def scale_up_deployment(self, deployment_name: str):
        """Scale up deployment """
        current_replicas = len(self.faas.get_replicas(deployment_name))
        
        logger.info(f"🔍 BEFORE scaling: {deployment_name} has {current_replicas} replicas")
        
        if current_replicas >= self.max_replicas:
            logger.warning(f"❌ Cannot scale {deployment_name}: already at max replicas")
            return
        
        selected_node = self.select_node_for_scaling(deployment_name)
        
        if selected_node:
            logger.info(f"🎯 Selected node: {selected_node.name} for {deployment_name}")
            
            # Record BEFORE scaling
            self.record_scaling_decision(deployment_name, "scale_up", current_replicas + 1, 
                                       selected_node, self.get_current_load(deployment_name), 
                                       self.get_average_response_time(deployment_name))
            
            try:
                # Perform actual scaling with error checking
                logger.info(f"🚀 Executing faas.scale_up({deployment_name}, 1)")
                yield from self.faas.scale_up(deployment_name, 1)
                
                # Check if it actually worked
                new_replicas = len(self.faas.get_replicas(deployment_name))
                logger.info(f"✅ AFTER scaling: {deployment_name} now has {new_replicas} replicas")
                
                if new_replicas == current_replicas:
                    logger.error(f"❌ SCALING FAILED: Replica count unchanged!")
                
            except Exception as e:
                logger.error(f"❌ SCALING ERROR: {e}")
                
        else:
            logger.warning(f"❌ No node selected for {deployment_name}")    
    def scale_down_deployment(self, deployment_name: str):
        """Scale down deployment"""
        current_replicas = len(self.faas.get_replicas(deployment_name))
        
        if current_replicas <= self.min_replicas:
            reason = f"Already at min replicas ({self.min_replicas})"
            current_load = self.get_current_load(deployment_name)
            avg_response_time = self.get_average_response_time(deployment_name)
            self.record_scaling_decision(deployment_name, "rejected_scale_down", 
                                       current_replicas, None, current_load, 
                                       avg_response_time, reason)
            logger.debug(f"⚠️  {deployment_name} {reason}")
            return
        
        # Get current metrics for logging
        current_load = self.get_current_load(deployment_name)
        avg_response_time = self.get_average_response_time(deployment_name)
        
        logger.info(f"🔽 {self.strategy_name}: Scaling down {deployment_name}")
        
        self.record_scaling_decision(deployment_name, "scale_down", current_replicas - 1, 
                                   None, current_load, avg_response_time,
                                   f"Low load: {current_load:.1f} RPS, Response: {avg_response_time:.1f}ms")
        
        # Perform actual scaling
        yield from self.faas.scale_down(deployment_name, 1)
        
        self.scaling_history[deployment_name] = current_replicas - 1
    
    def get_current_load(self, deployment_name: str) -> float:
        """Get REAL current RPS load using FaaS system's internal metrics"""
        try:
            # Method 1: Use FaaS system's internal invocation tracking
            # The DefaultFaasSystem tracks invocations in self.env.metrics.invocations
            current_total_invocations = self.env.metrics.invocations.get(deployment_name, 0)
            
            # Get previous count from our tracking
            if not hasattr(self, '_previous_invocations'):
                self._previous_invocations = {}
            
            previous_count = self._previous_invocations.get(deployment_name, 0)
            
            # Calculate invocations since last check
            new_invocations = current_total_invocations - previous_count
            
            # Update our tracking
            self._previous_invocations[deployment_name] = current_total_invocations
            
            # Calculate RPS based on our scaling interval
            rps = new_invocations / self.scaling_interval if self.scaling_interval > 0 else 0.0
            
            logger.debug(f"📊 Load for {deployment_name}: {rps:.2f} RPS "
                        f"({new_invocations} new invocations in {self.scaling_interval}s)")
            return rps
            
        except Exception as e:
            logger.error(f"❌ Error getting real load from FaaS system: {e}")
            return 0.0
    def get_average_response_time(self, deployment_name: str) -> float:
        """Get average response time from FETs DataFrame with detailed analysis"""
        try:
            # Cache DataFrame extraction (update every 30 seconds for performance)
            if not hasattr(self, '_fets_cache_time') or (self.env.now - self._fets_cache_time) > 30.0:
                self._fets_df_cache = self.env.metrics.extract_dataframe("fets")
                self._fets_cache_time = self.env.now
                logger.debug(f"📊 Refreshed FETs DataFrame cache at t={self.env.now:.1f}s")
            
            fets_df = self._fets_df_cache
            
            if fets_df is None or fets_df.empty:
                logger.debug(f"📊 No FET data available for {deployment_name}")
                return 0.0
            
            # Filter for the last 5 seconds window
            current_time = self.env.now
            window_start = current_time - 5.0  # 5 second window
            
            # Filter by time window and deployment name
            recent_fets = fets_df[
                (fets_df['t_fet_start'] >= window_start) &
                (fets_df['t_fet_start'] <= current_time) &
                (fets_df['function_name'].str.contains(deployment_name.replace('_', '-'), case=False, na=False))
            ]
            
            if recent_fets.empty:
                logger.debug(f"📊 No recent FET data for {deployment_name} in 5s window")
                return 0.0
            
            # Calculate all time components (in milliseconds)
            execution_times = (recent_fets['t_fet_end'] - recent_fets['t_fet_start']) * 1000
            wait_times = (recent_fets['t_wait_end'] - recent_fets['t_wait_start']) * 1000
            total_response_times = execution_times + wait_times
            
            # Get statistics
            avg_response_time = total_response_times.mean()
            avg_execution_time = execution_times.mean()
            avg_wait_time = wait_times.mean()
            
            # Calculate wait time percentage
            wait_percentage = (avg_wait_time / avg_response_time * 100) if avg_response_time > 0 else 0
            
            # Check for queueing issues (high wait times)
            high_wait_threshold = 100  # ms
            high_wait_count = (wait_times > high_wait_threshold).sum()
            
            logger.debug(f"📊 REAL metrics for {deployment_name}:")
            logger.debug(f"   Total response: {avg_response_time:.1f}ms")
            logger.debug(f"   Execution: {avg_execution_time:.1f}ms ({100-wait_percentage:.1f}%)")
            logger.debug(f"   Wait: {avg_wait_time:.1f}ms ({wait_percentage:.1f}%)")
            logger.debug(f"   Samples: {len(recent_fets)} in 10s window")
            
            if high_wait_count > 0:
                logger.debug(f"   ⚠️  {high_wait_count}/{len(recent_fets)} requests had high wait times (>{high_wait_threshold}ms)")
            
            # Store detailed metrics for strategy use
            if not hasattr(self, '_detailed_metrics'):
                self._detailed_metrics = {}
            
            # Append new metrics as a dict to a list for each deployment
            if deployment_name not in self._detailed_metrics:
                self._detailed_metrics[deployment_name] = []
            self._detailed_metrics[deployment_name].append({
                'avg_response_time': avg_response_time,
                'avg_execution_time': avg_execution_time,
                'avg_wait_time': avg_wait_time,
                'wait_percentage': wait_percentage,
                'high_wait_count': high_wait_count,
                'sample_count': len(recent_fets),
                'timestamp': current_time
            })
            detailed_data = {
                'deployment_name': deployment_name,
                'timestamp': current_time,
                'avg_response_time': avg_response_time,
                'avg_execution_time': avg_execution_time,
                'avg_wait_time': avg_wait_time,
                'wait_percentage': wait_percentage,
                'high_wait_count': high_wait_count,
                'sample_count': len(recent_fets)
            }
            
            # Log to environment metrics
            self.env.metrics.log('autoscaler_detailed_metrics', detailed_data)
            
            return avg_response_time
        except Exception as e:
            logger.error(f"❌ Error getting real response time from FETs: {e}")
            return 0.0
    

    def get_detailed_metrics_df(self) -> pd.DataFrame:
        """Return all detailed metrics as a DataFrame for analysis/export."""
        if not hasattr(self, '_detailed_metrics') or not self._detailed_metrics:
            return pd.DataFrame(columns=[
                'deployment_name', 'timestamp', 'avg_response_time', 'avg_execution_time',
                'avg_wait_time', 'wait_percentage', 'high_wait_count', 'sample_count'
            ])
        rows = []
        for deployment_name, metrics_list in self._detailed_metrics.items():
            for metrics in metrics_list:
                row = dict(metrics)
                row['deployment_name'] = deployment_name
                rows.append(row)
        df = pd.DataFrame(rows)
        df = df.sort_values(['deployment_name', 'timestamp'])
        return df
    
    def export_detailed_metrics_csv(self, filepath: str):
        """Export the detailed metrics DataFrame to a CSV file."""
        df = self.get_detailed_metrics_df()
        df.to_csv(filepath, index=False)
        logger.info(f"Exported detailed metrics to {filepath}")


    