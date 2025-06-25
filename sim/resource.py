from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sim.core import Environment
from sim.faas import FunctionReplica, FaasSystem, FunctionState


class ResourceUtilization:
    """Tracks resource consumption values for a single entity (function replica)"""

    __resources: Dict[str, float]

    def __init__(self):
        self.__resources = {}

    def put_resource(self, resource: str, value: float):
        """Add resource utilization (e.g. cpu, memory, gpu)"""
        if self.__resources.get(resource) is None:
            self.__resources[resource] = 0
        self.__resources[resource] += value

    def remove_resource(self, resource: str, value: float):
        """Release resources when function completes or scales down"""
        if self.__resources.get(resource) is None:
            self.__resources[resource] = 0
        self.__resources[resource] -= value

    def list_resources(self) -> Dict[str, float]:
        """Return copy of all resource utilization values"""
        return deepcopy(self.__resources)

    def copy(self) -> "ResourceUtilization":
        """Create independent copy of this utilization object"""
        util = ResourceUtilization()
        util.__resources = self.list_resources()
        return util

    def get_resource(self, resource) -> Optional[float]:
        """Retrieve specific resource usage (e.g. 'cpu')"""
        return self.__resources.get(resource)

    def is_empty(self) -> bool:
        """Check if no resources are being tracked"""
        return len(self.__resources) == 0


class NodeResourceUtilization:
    """Tracks all function replicas' resource usage on a single node"""

    # key is pod-name,   uniqueness allows for running same FunctionContainer multiple times on node
    __resources: Dict[str, ResourceUtilization]

    # associates the pod-name with its FunctionReplica
    __replicas: Dict[str, FunctionReplica]

    def __init__(self):
        self.__resources = {}
        self.__replicas = {}

    def put_resource(self, replica: FunctionReplica, resource: str, value: float):
        """Register resource usage for a specific function replica"""
        self.get_resource_utilization(replica).put_resource(resource, value)

    def remove_resource(self, replica: FunctionReplica, resource: str, value: float):
        """Release resources used by a specific function replica"""
        self.get_resource_utilization(replica).remove_resource(resource, value)

    def get_resource_utilization(self, replica: FunctionReplica) -> ResourceUtilization:
        """Get or create resource tracking for a function replica"""
        name = replica.pod.name
        util = self.__resources.get(name)
        if util is None:
            self.__resources[name] = ResourceUtilization()
            self.__replicas[name] = replica
            return self.__resources[name]
        else:
            return util

    def list_resource_utilization(
        self,
    ) -> List[Tuple[FunctionReplica, ResourceUtilization]]:
        """List all function replicas and their resource usage on this node"""
        functions = []
        for pod_name, utilization in self.__resources.items():
            replica = self.__replicas.get(pod_name)
            functions.append((replica, utilization))
        return functions

    @property
    def total_utilization(self) -> ResourceUtilization:
        """Calculate aggregated resource usage across all function replicas"""
        total = ResourceUtilization()
        for _, resource_utilization in self.list_resource_utilization():
            for resource, value in resource_utilization.list_resources().items():
                total.put_resource(resource, value)
        return total


class ResourceState:
    """Central registry tracking all resource utilization across all nodes"""

    node_resource_utilizations: Dict[str, NodeResourceUtilization]

    def __init__(self):
        self.node_resource_utilizations = {}

    def put_resource(
        self, function_replica: FunctionReplica, resource: str, value: float
    ):
        """Register resource usage for a function replica on its node"""
        node_name = function_replica.node.name
        node_resources = self.get_node_resource_utilization(node_name)
        node_resources.put_resource(function_replica, resource, value)

    def remove_resource(self, replica: "FunctionReplica", resource: str, value: float):
        """Release resources when a function completes execution"""
        node_name = replica.node.name
        self.get_node_resource_utilization(node_name).remove_resource(
            replica, resource, value
        )

    def get_resource_utilization(
        self, replica: "FunctionReplica"
    ) -> "ResourceUtilization":
        """Get current resource usage for a specific function replica"""
        node_name = replica.node.name
        return self.get_node_resource_utilization(node_name).get_resource_utilization(
            replica
        )

    def list_resource_utilization(
        self, node_name: str
    ) -> List[Tuple["FunctionReplica", "ResourceUtilization"]]:
        """List all function replicas and their resource usage on a node"""
        return self.get_node_resource_utilization(node_name).list_resource_utilization()

    def get_node_resource_utilization(
        self, node_name: str
    ) -> Optional[NodeResourceUtilization]:
        """Get or create resource tracking for a specific node"""
        node_resources = self.node_resource_utilizations.get(node_name)
        if node_resources is None:
            self.node_resource_utilizations[node_name] = NodeResourceUtilization()
            node_resources = self.node_resource_utilizations[node_name]
        return node_resources


@dataclass
class ResourceWindow:
    """Snapshot of resource usage at a specific point in time"""

    replica: FunctionReplica
    resources: Dict[str, float]
    time: float


class MetricsServer:
    """
    Stores time-series resource usage data for later analysis and querying
    """

    def __init__(self):
        # Nested dictionary: node -> pod -> list of resource windows
        self._windows = defaultdict(lambda: defaultdict(list))

    def put(self, window: ResourceWindow):
        """Add a new resource usage snapshot to the timeseries"""
        node = window.replica.node.name
        pod = window.replica.pod.name
        self._windows[node][pod].append(window)

    def get_average_cpu_utilization(
        self, fn_replica: FunctionReplica, window_start: float, window_end: float
    ) -> float:
        """Calculate average CPU usage over a time window as fraction of capacity"""
        utilization = self.get_average_resource_utilization(
            fn_replica, "cpu", window_start, window_end
        )
        millis = fn_replica.node.capacity.cpu_millis
        return utilization / millis

    def get_average_resource_utilization(
        self,
        fn_replica: FunctionReplica,
        resource: str,
        window_start: float,
        window_end: float,
    ) -> float:
        """Calculate average usage of a specific resource over a time window"""
        node = fn_replica.node.name
        pod = fn_replica.pod.name
        windows: List[ResourceWindow] = self._windows.get(node, {}).get(pod, [])
        if len(windows) == 0:
            return 0
        average_windows = []

        # Find all windows within the time range (searching backward for efficiency)
        for window in reversed(windows):
            if window.time <= window_end:
                if window.time < window_start:
                    break
                average_windows.append(window)

        # Calculate average of resource values across all windows
        return np.mean(list(map(lambda l: l.resources[resource], average_windows)))


class ResourceMonitor:
    """Background process that periodically samples and records resource utilization"""

    def __init__(self, env: Environment, reconcile_interval: int, logging=True):
        self.env = env
        self.reconcile_interval = (
            reconcile_interval  # How often to sample resource usage
        )
        self.metric_server: MetricsServer = env.metrics_server
        self.logging = logging

    def run(self):
        """SimPy process that runs throughout simulation, collecting resource metrics"""
        faas: FaasSystem = self.env.faas
        while True:
            # Wait for next collection interval
            yield self.env.timeout(self.reconcile_interval)
            now = self.env.now

            # Iterate through all running function replicas
            for deployment in faas.get_deployments():
                for replica in faas.get_replicas(
                    deployment.name, FunctionState.RUNNING
                ):
                    # Get current resource usage for this replica
                    utilization = self.env.resource_state.get_resource_utilization(
                        replica
                    )
                    if utilization.is_empty():
                        continue

                    # Log metrics and add to metrics server timeseries
                    if self.logging:
                        self.env.metrics.log_function_resource_utilization(
                            replica, utilization
                        )
                    self.metric_server.put(
                        ResourceWindow(replica, utilization.list_resources(), now)
                    )
