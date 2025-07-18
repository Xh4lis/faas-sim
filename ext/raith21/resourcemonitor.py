from typing import Dict, List

import numpy as np

from ext.raith21.functionsim import FunctionCall
from sim.core import Environment
from sim.faas import FaasSystem, FunctionState
from sim.oracle.oracle import ResourceOracle
from sim.resource import ResourceWindow


class Raith21ResourceMonitor:

    def __init__(self, env: Environment, resource_oracle: ResourceOracle):
        self.env = env
        self.resource_oracle = resource_oracle
        self.metric_server = env.metrics_server

    def run(self):
        faas: FaasSystem = self.env.faas
        while True:
            start_ts = self.env.now
            yield self.env.timeout(1)
            end_ts = self.env.now
            
            # calculate resources over function replica resources and save in metric_server
            call_cache: Dict[str, List[FunctionCall]] = {}
            for function_deployment in faas.get_deployments():
                for replica in faas.get_replicas(
                    function_deployment.name, FunctionState.RUNNING
                ):
                    node_name = replica.node.name
                    
                    # Skip infrastructure nodes
                    if any(keyword in node_name.lower() for keyword in ['link', 'switch', 'registry']):
                        continue
                    
                    calls = call_cache.get(node_name, None)
                    if calls is None:
                        calls = replica.node.get_calls_in_timeframe(start_ts, end_ts)
                        call_cache[node_name] = calls
                    
                    trace_execution_durations = []
                    replica_usage = self.resource_oracle.get_resources(
                        node_name, replica.function.image
                    )
                    
                    for call in calls:
                        if call.replica.pod.name == replica.pod.name:
                            last_start = (
                                start_ts if start_ts >= call.start else call.start
                            )

                            if call.end is not None:
                                first_end = end_ts if end_ts <= call.end else call.end
                            else:
                                first_end = end_ts

                            overlap = first_end - last_start
                            trace_execution_durations.append(overlap)
                    
                    if len(trace_execution_durations) == 0:
                        window = ResourceWindow(replica, 0)
                        
                        replica_cpu = 0.0
                        replica_memory = 0.0
                        replica_gpu = 0.0
                        
                    else:
                        sum_duration = np.sum(trace_execution_durations)
                        cpu_usage = sum_duration * replica_usage["cpu"]
                        cpu_usage_capped = min(1, cpu_usage)
                        
                        window = ResourceWindow(replica, cpu_usage_capped)
                        
                        replica_cpu = cpu_usage_capped
                        replica_memory = min(1.0, sum_duration * replica_usage.get("ram", 0.0))
                        replica_gpu = min(1.0, sum_duration * replica_usage.get("gpu", 0.0))
                    
                    self.metric_server.put(window)
                    
                    if hasattr(self.env, 'resource_state') and self.env.resource_state:
                        self.env.resource_state.put_resource(replica, 'cpu', replica_cpu)
                        self.env.resource_state.put_resource(replica, 'memory', replica_memory)
                        if replica_gpu > 0:
                            self.env.resource_state.put_resource(replica, 'gpu', replica_gpu)
                    
                    if hasattr(self.env, 'metrics') and self.env.metrics:
                        # Create ResourceUtilization for metrics logging
                        from sim.resource import ResourceUtilization
                        replica_util = ResourceUtilization()
                        replica_util.put_resource('cpu', replica_cpu)
                        replica_util.put_resource('memory', replica_memory)
                        if replica_gpu > 0:
                            replica_util.put_resource('gpu', replica_gpu)
                        
                        self.env.metrics.log_function_resource_utilization(replica, replica_util)