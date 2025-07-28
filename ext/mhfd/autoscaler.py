import logging
import os
from typing import Dict, Any, Optional

from ext.mhfd.scaling.strategies.standard_first_fit_packer import StandardFirstFitBinPacker
from ext.mhfd.scaling.strategies.lplt import LowPowerLongTimeBinPacker
from ext.mhfd.scaling.strategies.hpst import HighPerformanceShortTimeBinPacker

logger = logging.getLogger(__name__)

class Autoscaler:
    """
    Autoscaler for heterogeneous edge computing environments
    Coordinates different bin-packing strategies for serverless function placement
    """
    
    def __init__(self, env, faas_system, power_oracle, strategy_name: str = "basic"):
        self.env = env
        self.faas = faas_system
        self.power_oracle = power_oracle
        self.strategy_name = strategy_name
        
        # Initialize the selected strategy
        self.strategy = self._create_strategy(strategy_name)
        
        # Energy tracking
        self.total_energy_consumed = 0.0
        self.energy_efficiency_history = []
        
        logger.info(f"🚀 Autoscaler initialized with {strategy_name} strategy")
    def __getattr__(self, name):
        """Delegate method calls to the underlying strategy"""
        if hasattr(self.strategy, name):
            return getattr(self.strategy, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _create_strategy(self, strategy_name: str):
        """Create the appropriate scaling strategy"""
        strategies = {
            'basic': StandardFirstFitBinPacker,
            'power': LowPowerLongTimeBinPacker,
            'performance': HighPerformanceShortTimeBinPacker
        }
        
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
        
        return strategies[strategy_name](self.env, self.faas, self.power_oracle)
    
    def run(self):
        """Main autoscaling process - delegates to strategy"""
        logger.info(f"🔄 Starting energy-aware autoscaling with {self.strategy_name} strategy")
        
        # Start the strategy's scaling process
        yield from self.strategy.run()
    
    def get_energy_efficiency_metrics(self) -> Dict[str, float]:
        """Get current energy efficiency metrics"""
        total_energy = self.env.power_metrics.get_total_energy() if hasattr(self.env, 'power_metrics') else 0.0
        
        # Calculate total completed invocations
        total_invocations = 0
        if hasattr(self.env, 'metrics'):
            try:
                # This would need to be implemented based on your metrics structure
                for deployment_name in self.faas.deployments:
                    # Get invocation count from metrics
                    total_invocations += getattr(self.env.metrics, 'total_invocations', {}).get(deployment_name, 0)
            except:
                total_invocations = 1  # Avoid division by zero
        
        energy_per_invocation = total_energy / max(total_invocations, 1)
        
        return {
            'strategy': self.strategy_name,
            'total_energy_joules': total_energy,
            'total_energy_wh': total_energy / 3600.0,
            'total_invocations': total_invocations,
            'energy_per_invocation_joules': energy_per_invocation,
            'energy_efficiency_score': 1.0 / max(energy_per_invocation, 0.001)  # Higher is better
        }
    
    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the current strategy"""
        return {
            'strategy_name': self.strategy_name,
            'scaling_decisions': getattr(self.strategy, 'scaling_history', {}),
            'energy_metrics': self.get_energy_efficiency_metrics(),
            'active_replicas': self._count_active_replicas(),
            'node_utilization': self._get_node_utilization_summary()
        }
    
    def _count_active_replicas(self) -> Dict[str, int]:
        """Count active replicas per deployment"""
        replica_counts = {}
        for deployment_name in self.faas.deployments:
            replica_counts[deployment_name] = len(self.faas.get_replicas(deployment_name))
        return replica_counts
    
    def _get_node_utilization_summary(self) -> Dict[str, Dict[str, float]]:
        """Get utilization summary per node type"""
        from ext.mhfd.power import extract_node_type, get_current_utilization
        
        node_util_summary = {}
        node_type_counts = {}
        
        for node in self.env.topology.get_nodes():
            if hasattr(node, 'capacity'):
                node_type = extract_node_type(node.name)
                utilization = get_current_utilization(self.env, node.name)
                
                if node_type not in node_util_summary:
                    node_util_summary[node_type] = {'cpu': 0.0, 'memory': 0.0, 'gpu': 0.0}
                    node_type_counts[node_type] = 0
                
                node_util_summary[node_type]['cpu'] += utilization.get('cpu', 0.0)
                node_util_summary[node_type]['memory'] += utilization.get('memory', 0.0)
                node_util_summary[node_type]['gpu'] += utilization.get('gpu', 0.0)
                node_type_counts[node_type] += 1
        
        # Average utilization per node type
        for node_type in node_util_summary:
            count = node_type_counts[node_type]
            if count > 0:
                node_util_summary[node_type]['cpu'] /= count
                node_util_summary[node_type]['memory'] /= count
                node_util_summary[node_type]['gpu'] /= count
        
        return node_util_summary


def create_heterogeneous_edge_autoscaler(env, faas_system, power_oracle, strategy: str = None) -> Autoscaler:
    """Factory function to create heterogeneous edge autoscaler"""

    # Determine strategy from environment variable or parameter
    if strategy is None:
        strategy = os.getenv('SCALING_STRATEGY', 'basic')
    
    logger.info(f"🏭 Creating energy-aware autoscaler with {strategy} strategy")
    
    return Autoscaler(env, faas_system, power_oracle, strategy)


def compare_scaling_strategies(results_dir: str = "results"):
    """Compare results from different scaling strategies"""
    import pandas as pd
    import glob
    
    strategy_results = {}
    
    # Look for results from different strategies
    for strategy in ['basic', 'power', 'performance']:
        result_files = glob.glob(f"{results_dir}/*_{strategy}_*.csv")
        if result_files:
            strategy_results[strategy] = result_files
    
    if not strategy_results:
        logger.warning("No strategy results found for comparison")
        return
    
    logger.info("📊 SCALING STRATEGY COMPARISON")
    logger.info("=" * 50)
    
    for strategy, files in strategy_results.items():
        logger.info(f"\n{strategy.upper()} STRATEGY:")
        logger.info(f"  Result files: {len(files)}")
        
        # Try to load and summarize results
        try:
            for file in files:
                if 'power' in file:
                    df = pd.read_csv(file)
                    if not df.empty:
                        avg_power = df['power_watts'].mean()
                        total_energy = df['power_watts'].sum() * 30 / 3600  # Approximate Wh
                        logger.info(f"  Average power: {avg_power:.2f} W")
                        logger.info(f"  Total energy: {total_energy:.2f} Wh")
                elif 'invocation' in file:
                    df = pd.read_csv(file)
                    if not df.empty:
                        avg_response_time = (df['wait_time'] + df['exec_time']).mean()
                        total_invocations = len(df)
                        logger.info(f"  Average response time: {avg_response_time:.2f} ms")
                        logger.info(f"  Total invocations: {total_invocations}")
        except Exception as e:
            logger.warning(f"  Could not analyze {strategy} results: {e}")