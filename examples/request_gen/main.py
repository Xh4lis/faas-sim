import logging
from typing import List

import ether.scenarios.urbansensing as scenario
from skippy.core.utils import parse_size_string

from sim import docker
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDeployment, FunctionRequest, Function, FunctionImage, ScalingConfiguration, \
    DeploymentRanking, FunctionContainer, KubernetesResourceConfiguration
from sim.faassim import Simulation
from sim.topology import Topology

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # a topology holds the cluster configuration and network topology
    topology = traffic_detection_topology()

    # a benchmark is a simpy process that sets up the runtime system and creates workload
    benchmark = TrafficDetectionBenchmark()

    # a simulation runs until the benchmark process terminates
    sim = Simulation(topology, benchmark)
    sim.run()


def traffic_detection_topology() -> Topology:
    # The urban sensing scenario works well for traffic detection since it models
    # a city with different layers of computing resources
    t = Topology()
    scenario.UrbanSensingScenario().materialize(t)
    t.init_docker_registry()
    
    # Rename nodes to better reflect a traffic monitoring system
    node_mapping = {
        'cloud': 'traffic_management_center',
        'isp': 'district_hub',
        'pc': 'intersection_controller',
        'rpi3': 'roadside_monitor',
        'rpi4': 'sensor_gateway'
    }
    
    # Adjust node names for clarity (not actually changing the nodes)
    for node in t.nodes:
        node_type = node.split('_')[0]
        if node_type in node_mapping:
            logger.info(f"Node {node} represents a {node_mapping[node_type]} in our traffic system")
    
    return t


class TrafficDetectionBenchmark(Benchmark):

    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        # Register container images, reusing existing ones but with traffic-appropriate roles
        # For traffic detection - GPU version (similar to resnet50-inference-gpu)
        containers.put(ImageProperties('traffic-detection-gpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('traffic-detection-gpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('traffic-detection-gpu', parse_size_string('56M'), arch='aarch64'))
        
        # For traffic detection - CPU version (similar to resnet50-inference-cpu)
        containers.put(ImageProperties('traffic-detection-cpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('traffic-detection-cpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('traffic-detection-cpu', parse_size_string('56M'), arch='aarch64'))
        
        # For data aggregation (similar to python-pi-cpu)
        containers.put(ImageProperties('data-aggregator', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('data-aggregator', parse_size_string('58M'), arch='x86'))
        containers.put(ImageProperties('data-aggregator', parse_size_string('58M'), arch='aarch64'))
        
        # For traffic flow analysis (based on python-pi-cpu but with different role)
        containers.put(ImageProperties('traffic-flow-analyzer', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('traffic-flow-analyzer', parse_size_string('58M'), arch='x86'))
        containers.put(ImageProperties('traffic-flow-analyzer', parse_size_string('58M'), arch='aarch64'))

        # Log all the images in the container registry
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        # Deploy functions
        deployments = self.prepare_deployments()

        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # Block until replicas become available
        logger.info('waiting for replicas')
        yield env.process(env.faas.poll_available_replica('traffic-detection'))
        yield env.process(env.faas.poll_available_replica('data-aggregation'))
        yield env.process(env.faas.poll_available_replica('traffic-analysis'))

        # Run workload - simulating a realistic traffic monitoring pattern
        ps = []
        
        # Continuous monitoring - regular vehicle detection (higher volume)
        logger.info('executing 20 traffic-detection requests (continuous monitoring)')
        for i in range(20):
            ps.append(env.process(env.faas.invoke(FunctionRequest('traffic-detection'))))
        
        # Sensor data aggregation - collecting from various sensors
        logger.info('executing 15 data-aggregation requests (sensor collection)')
        for i in range(15):
            ps.append(env.process(env.faas.invoke(FunctionRequest('data-aggregation'))))
        
        # Traffic analysis - periodic analysis of trends and patterns
        logger.info('executing 8 traffic-analysis requests (trend analysis)')
        for i in range(8):
            ps.append(env.process(env.faas.invoke(FunctionRequest('traffic-analysis'))))
        
        # Event-triggered incidents - spikes during congestion or incidents 
        logger.info('executing 5 high-priority traffic-detection requests (incident detection)')
        high_priority_req = FunctionRequest('traffic-detection')
        high_priority_req.labels = {'priority': 'high'}
        for i in range(5):
            ps.append(env.process(env.faas.invoke(high_priority_req)))

        # Wait for all invocation processes to finish
        for p in ps:
            yield p

    def prepare_deployments(self) -> List[FunctionDeployment]:
        traffic_detection_fd = self.prepare_traffic_detection_deployment()
        data_aggregation_fd = self.prepare_data_aggregation_deployment()
        traffic_analysis_fd = self.prepare_traffic_analysis_deployment()

        return [traffic_detection_fd, data_aggregation_fd, traffic_analysis_fd]

    def prepare_traffic_detection_deployment(self):
        # Design time - define the function with CPU and GPU options
        func_name = 'traffic-detection'
        cpu_image = FunctionImage(image='traffic-detection-cpu')
        gpu_image = FunctionImage(image='traffic-detection-gpu')
        
        traffic_detection_fn = Function(func_name, fn_images=[gpu_image, cpu_image])

        # Run time - configure resource requirements
        
        # Default container for CPU version
        cpu_container = FunctionContainer(cpu_image)
        
        # GPU version with higher resource allocation for video processing
        gpu_resource_config = KubernetesResourceConfiguration.create_from_str(
            cpu='500m',    # Higher CPU for video processing
            memory='2048Mi'  # More memory for frame buffers
        )
        gpu_container = FunctionContainer(gpu_image, resource_config=gpu_resource_config)
        
        # Create deployment with GPU preferred over CPU
        deployment = FunctionDeployment(
            traffic_detection_fn,
            [gpu_container, cpu_container],
            ScalingConfiguration(scale_min=2, scale_max=10),  # Allow scaling for busy periods
            DeploymentRanking(['traffic-detection-gpu', 'traffic-detection-cpu'])
        )

        return deployment

    def prepare_data_aggregation_deployment(self):
        # Design time - lightweight function for collecting and processing sensor data
        func_name = 'data-aggregation'
        image = FunctionImage(image='data-aggregator')
        
        data_agg_fn = Function(func_name, fn_images=[image])

        # Run time - minimal resource requirements
        container = FunctionContainer(image)
        
        # Create deployment with modest scaling configuration
        deployment = FunctionDeployment(
            data_agg_fn,
            [container],
            ScalingConfiguration(scale_min=1, scale_max=5)
        )

        return deployment

    def prepare_traffic_analysis_deployment(self):
        # Design time - analytical function for traffic pattern analysis
        func_name = 'traffic-analysis'
        image = FunctionImage(image='traffic-flow-analyzer')
        
        traffic_analysis_fn = Function(func_name, fn_images=[image])

        # Run time - needs more memory for data processing
        resource_config = KubernetesResourceConfiguration.create_from_str(
            cpu='200m',
            memory='1024Mi'  # More memory for analytics
        )
        container = FunctionContainer(image, resource_config=resource_config)
        
        # Create deployment with limited scaling (analytical tasks)
        deployment = FunctionDeployment(
            traffic_analysis_fn,
            [container],
            ScalingConfiguration(scale_min=1, scale_max=3)
        )

        return deployment


if __name__ == '__main__':
    main()