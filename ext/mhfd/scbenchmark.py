from ext.raith21.benchmark.constant import ConstantBenchmark
from ext.raith21.fet import ai_execution_time_distributions
from ext.raith21.oracles import Raith21FetOracle, Raith21ResourceOracle
from ext.raith21.resources import ai_resources_per_node_image
from ext.mhfd.deployments import create_smart_city_deployments, create_custom_smart_city_deployments
from sim.requestgen import expovariate_arrival_profile, constant_rps_profile


class SmartCityConstantBenchmark(ConstantBenchmark):
    """
    Enhanced ConstantBenchmark that uses smart city deployments and realistic arrival profiles.
    """
    
    def __init__(self, profile: str, duration: int, rps=200, 
                 scenario: str = "default", custom_counts=None, model_folder=None):
        """
        Initialize Smart City Constant Benchmark.
        
        Args:
            profile: Workload profile ("mixed", "ai", "service")
            duration: Simulation duration
            rps: Total requests per second
            scenario: Smart city scenario ("default", "intensive", "distributed", "custom")
            custom_counts: Custom instance counts for "custom" scenario
            model_folder: Optional model folder for degradation
        """
        self.scenario = scenario
        self.custom_counts = custom_counts
        
        # Call parent constructor but we'll override deployments
        super().__init__(profile, duration, rps, model_folder)
        
        # Replace deployments with smart city deployments
        fet_oracle = Raith21FetOracle(ai_execution_time_distributions)
        resource_oracle = Raith21ResourceOracle(ai_resources_per_node_image)
        
        if scenario == "custom" and custom_counts:
            smart_deployments = create_custom_smart_city_deployments(
                fet_oracle, resource_oracle, custom_counts
            )
        else:
            smart_deployments = create_smart_city_deployments(
                fet_oracle, resource_oracle, scenario
            )
        
        # Replace the original deployments
        self.deployments = smart_deployments
        self.deployments_per_name = {dep.name: dep for dep in smart_deployments}
        
        # Clear original arrival profiles - will be set in setup_profile
        self.arrival_profiles = {}

    def set_mixed_profiles(self):
        """Set smart city mixed workload arrival profiles."""
        self._set_smart_city_mixed_profiles()

    def _set_smart_city_mixed_profiles(self):
        """Set arrival profiles optimized for smart city mixed workload."""
        
        # Smart city zone-based RPS allocation
        zone_multipliers = {
            'downtown': 1.5,      # High activity downtown
            'commercial': 1.3,    # High activity commercial
            'highway': 1.4,       # High traffic on highways
            'airport': 1.2,       # Moderate high activity
            'stadium': 1.6,       # Very high during events
            'industrial': 1.1,    # Moderate activity
            'port': 1.0,          # Steady activity
            'hospital': 0.9,      # Lower but critical
            'university': 1.0,    # Moderate activity
            'residential': 0.7,   # Lower residential activity
            'suburb': 0.6,        # Low suburban activity
            'park': 0.5,          # Low park activity
        }
        
        # Function type base RPS allocation (percentage of total RPS)
        function_base_rps = {
            'resnet50-inference': 0.25,     # Reduce from 35% to 25%
            'speech-inference': 0.25,       # Increase from 20% to 25%
            'resnet50-preprocessing': 0.20, # Reduce from 25% to 20%
            'resnet50-training': 0.10,      # Increase from 5% to 10%
            'python-pi': 0.15,              # 15% for edge computing
            'fio': 0.05,                    # 5% for I/O monitoring
        }
                
        total_rps = self.rps
        deployment_rps = {}
        
        # Calculate RPS for each deployment
        for deployment in self.deployments:
            func_name = deployment.name
            
            # Extract function type and zone
            parts = func_name.split('-')
            if len(parts) >= 2:
                base_func = f"{parts[0]}-{parts[1]}"
                zone = parts[2] if len(parts) > 2 else 'default'
            else:
                base_func = func_name
                zone = 'default'
            
            # Get base RPS for this function type
            base_percentage = function_base_rps.get(base_func, 0.05)
            base_rps = total_rps * base_percentage
            
            # Apply zone multiplier
            zone_multiplier = zone_multipliers.get(zone, 1.0)
            
            # Calculate instances of this function type
            same_function_deployments = [d for d in self.deployments 
                                       if d.name.startswith(base_func)]
            instance_count = len(same_function_deployments)
            
            # Final RPS calculation
            final_rps = max(1, int((base_rps * zone_multiplier) / instance_count))
            deployment_rps[func_name] = final_rps
        
        # Set arrival profiles
        for deployment in self.deployments:
            func_name = deployment.name
            rps = deployment_rps.get(func_name, 1)
            
            self.arrival_profiles[func_name] = expovariate_arrival_profile(
                constant_rps_profile(rps)
            )
        
        # Print distribution summary
        print(f"\n=== SMART CITY MIXED PROFILES ===")
        print(f"Scenario: {self.scenario}")
        print(f"Total deployments: {len(self.deployments)}")
        
        total_assigned = sum(deployment_rps.values())
        print(f"Total RPS assigned: {total_assigned} (target: {total_rps})")
        
        # Group by function type for summary
        function_summary = {}
        for func_name, rps in deployment_rps.items():
            base_func = '-'.join(func_name.split('-')[:2])
            if base_func not in function_summary:
                function_summary[base_func] = {'count': 0, 'total_rps': 0}
            function_summary[base_func]['count'] += 1
            function_summary[base_func]['total_rps'] += rps
        
        for func_type, stats in function_summary.items():
            avg_rps = stats['total_rps'] / stats['count']
            print(f"  {func_type}: {stats['count']} instances, "
                  f"{stats['total_rps']} total RPS, {avg_rps:.1f} avg RPS")


# Convenience function for easy creation
def create_smart_city_constant_benchmark(
    duration: int = 500,
    total_rps: int = 1000,
    scenario: str = "default",
    custom_counts: dict = None
):
    """Create a smart city constant benchmark with sensible defaults."""
    return SmartCityConstantBenchmark(
        profile="mixed",
        duration=duration,
        rps=total_rps,
        scenario=scenario,
        custom_counts=custom_counts
    )