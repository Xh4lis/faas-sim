import logging
import random
from typing import List

from sim import docker
from skippy.core.utils import parse_size_string
from srds import ParameterizedDistribution

import ether.scenarios.urbansensing as scenario
from sim.benchmark import Benchmark
from sim.core import Environment
from sim.docker import ImageProperties
from sim.faas import FunctionDeployment, FunctionRequest, Function, FunctionImage, ScalingConfiguration, \
    DeploymentRanking, FunctionContainer, KubernetesResourceConfiguration, FunctionSimulator, SimulatorFactory, \
    FunctionReplica
from sim.faassim import Simulation
from sim.topology import Topology

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    # a topology holds the cluster configuration and network topology
    topology = example_topology()

    # Use the enhanced benchmark
    benchmark = EnhancedBenchmark()

    sim = Simulation(topology, benchmark)
    
    # Use custom simulator factory for realistic performance
    sim.create_simulator_factory = CitySimulatorFactory
    
    sim.run()


def example_topology() -> Topology:
    t = Topology()
    
    # Create a scenario with custom parameters
    urban_scenario = scenario.UrbanSensingScenario(
        num_cells=5,                                       # More neighborhoods
        cell_density=ParameterizedDistribution.lognorm((2.079, 0.001)), 
        cloudlet_size=(4, 3)                               # 4 servers per rack, 3 racks
    )
    
    urban_scenario.materialize(t)
    t.init_docker_registry()

    return t


class RealisticFunctionSimulator(FunctionSimulator):
    """Base class for realistic function simulators"""
    
    def __init__(self, replica):
        self.replica = replica
        self.default_exec_time = 0.01  # 10ms
        self.default_cpu_util = 10     # 10%
    
    def deploy(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)
        
    def startup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)
        
    def setup(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)
        
    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        """Execute the function and return the execution time and CPU utilization"""
        exec_time, cpu_util = self._calculate_execution_metrics(request)
        yield env.timeout(exec_time)
        return
        
    def teardown(self, env: Environment, replica: FunctionReplica):
        yield env.timeout(0)
        
    def _calculate_execution_metrics(self, request):
        """Calculate realistic execution metrics"""
        # Add variability to execution time (45-65ms)
        exec_time = random.uniform(0.045, 0.065)
        
        # CPU utilization varies between 60-80%
        cpu_util = random.uniform(60, 80)
        
        # Add size-based variability if available
        if hasattr(request, 'size') and request.size:
            # Scale execution time based on size (iterations)
            iter_scale = request.size / 1000
            exec_time = exec_time * (1 + iter_scale)
            
        return exec_time, cpu_util

class PythonPiSimulator(RealisticFunctionSimulator):
    """Realistic simulator for Python Pi calculation"""
    
    def __init__(self, replica):
        super().__init__(replica)
        # Pi calculation is CPU intensive but not extremely heavy
        self.default_exec_time = 0.05  # 50ms
        self.default_cpu_util = 70     # 70% CPU usage
    
    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        """Execute the Python Pi calculation"""
        exec_time, cpu_util = self._calculate_execution_metrics(request)
        logger.info(f"PythonPi execution time: {exec_time*1000:.2f}ms, CPU: {cpu_util:.1f}%")
        yield env.timeout(exec_time)
        return
        
    def _calculate_execution_metrics(self, request):
        """Calculate realistic execution metrics for Pi calculation"""
        # Add variability to execution time (45-65ms)
        exec_time = random.uniform(0.045, 0.065)
        
        # CPU utilization varies between 60-80%
        cpu_util = random.uniform(60, 80)
        
        # Use size attribute for iteration scaling
        if hasattr(request, 'size') and request.size:
            # Scale execution time based on size (iterations)
            iter_scale = request.size / 1000
            exec_time = exec_time * (1 + iter_scale)
            
        return exec_time, cpu_util
    

class ResNet50Simulator(RealisticFunctionSimulator):
    """Realistic simulator for ResNet50 inference"""
    
    def __init__(self, replica):
        super().__init__(replica)
        self.is_gpu = replica and replica.image and 'gpu' in replica.image
        
        # Set baseline performance
        if self.is_gpu:
            # GPU version is significantly faster
            self.default_exec_time = 0.035  # 35ms on GPU
            self.default_cpu_util = 30      # 30% CPU on GPU version
            if replica and replica.node:
                logger.info(f"Using GPU version of ResNet50 on {replica.node.name}")
        else:
            # CPU version is much slower
            self.default_exec_time = 0.180  # 180ms on CPU
            self.default_cpu_util = 90      # 90% CPU utilization
            if replica and replica.node:
                logger.info(f"Using CPU version of ResNet50 on {replica.node.name}")
    
    def invoke(self, env: Environment, replica: FunctionReplica, request: FunctionRequest):
        """Execute the ResNet50 inference"""
        exec_time, cpu_util = self._calculate_execution_metrics(request)
        logger.info(f"ResNet50 execution time: {exec_time*1000:.2f}ms, CPU: {cpu_util:.1f}%")
        yield env.timeout(exec_time)
        return
        
    def _calculate_execution_metrics(self, request):
        """Calculate realistic execution metrics for ResNet50 inference"""
        # Base execution times with realistic values for ResNet50
        if self.is_gpu:
            # GPU execution time between 30-50ms
            exec_time = random.uniform(0.030, 0.050)
            # GPU version uses less CPU
            cpu_util = random.uniform(25, 40)
        else:
            # CPU execution time between 150-250ms
            exec_time = random.uniform(0.150, 0.250)
            # CPU version maxes out the CPU
            cpu_util = random.uniform(80, 95)
        
        # Use size attribute for image size and batch size
        if hasattr(request, 'size') and request.size:
            size_value = request.size
            
            # Interpret size value:
            # - Values < 1.5: small image (0.7x factor)
            # - Values 1.5-3.0: medium image (1.0x factor)
            # - Values > 3.0: large image (1.5x factor)
            
            if size_value < 1.5:
                # Small image
                exec_time *= 0.7
            elif size_value > 3.0:
                # Large image
                exec_time *= 1.5
                
            # If size > 10, interpret as batch processing
            if size_value > 10:
                batch_size = int(size_value / 10)
                if batch_size > 1:
                    # Batch processing is more efficient but still scales
                    exec_time = exec_time * (0.7 * batch_size)
        
        return exec_time, cpu_util

class CitySimulatorFactory(SimulatorFactory):
    """Factory that creates realistic function simulators"""
    
    def create(self, env: Environment, fn: FunctionContainer) -> FunctionSimulator:
        """Create an appropriate function simulator based on the container's function"""
        # First find the replica associated with this container
        if not hasattr(env, 'replicas_by_container'):
            # Create a lookup cache if it doesn't exist
            env.replicas_by_container = {}
            
        # Find the replica for this function container
        replica = None
        for fn_name, replicas in env.faas.replicas.items():
            for r in replicas:
                if r.container == fn:
                    replica = r
                    env.replicas_by_container[fn] = r
                    break
            if replica:
                break
        
        if not replica or not replica.function:
            # Create a basic simulator if we can't find the replica
            return RealisticFunctionSimulator(replica)
            
        function_name = replica.function.name
        
        if function_name == 'python-pi':
            return PythonPiSimulator(replica)
        elif function_name == 'resnet50-inference':
            return ResNet50Simulator(replica)
        
        # Default for other functions
        return RealisticFunctionSimulator(replica)

class EnhancedBenchmark(Benchmark):
    """A more realistic benchmark that creates significant load on the system"""
    
    def setup(self, env: Environment):
        containers: docker.ContainerRegistry = env.container_registry

        # Register the same container images as in the original benchmark
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='arm32'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='x86'))
        containers.put(ImageProperties('python-pi-cpu', parse_size_string('58M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-cpu', parse_size_string('56M'), arch='aarch64'))

        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='arm32'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='x86'))
        containers.put(ImageProperties('resnet50-inference-gpu', parse_size_string('56M'), arch='aarch64'))

        # Log registered images
        for name, tag_dict in containers.images.items():
            for tag, images in tag_dict.items():
                logger.info('%s, %s, %s', name, tag, images)

    def run(self, env: Environment):
        """Run a more intensive workload simulation"""
        # Deploy functions
        deployments = self.prepare_deployments()
        for deployment in deployments:
            yield from env.faas.deploy(deployment)

        # Wait for replicas to become available
        logger.info('Waiting for replicas to be available')
        yield env.process(env.faas.poll_available_replica('python-pi'))
        yield env.process(env.faas.poll_available_replica('resnet50-inference'))
        
        # Get topology nodes for network traffic simulation
        edge_nodes = [node for node in env.topology.nodes 
                     if isinstance(node, str) and "nuc_" in node]
        cloud_nodes = [node for node in env.topology.nodes 
                     if isinstance(node, str) and "server_" in node]
        
        if not edge_nodes or not cloud_nodes:
            logger.warning("Could not find edge or cloud nodes for network simulation")
            edge_nodes = ["edge_0"]
            cloud_nodes = ["server_0"]

        # Run multiple phases of the workload to simulate different traffic patterns
        yield env.process(self._run_light_load_phase(env, edge_nodes, cloud_nodes))
        yield env.process(self._run_medium_load_phase(env, edge_nodes, cloud_nodes))
        yield env.process(self._run_heavy_load_phase(env, edge_nodes, cloud_nodes))
        yield env.process(self._run_bursty_load_phase(env, edge_nodes, cloud_nodes))
        
        logger.info("Benchmark completed")

    def _run_light_load_phase(self, env, edge_nodes, cloud_nodes):
        """Simulate light load (occasional requests)"""
        logger.info("Starting LIGHT LOAD PHASE (60 seconds)")
        phase_end = env.now + 60  # 60 second phase
        
        while env.now < phase_end:
            # Generate 1 request every 2-5 seconds
            yield env.timeout(random.uniform(2, 5))
            
            # 70% chance of Python Pi, 30% chance of ResNet50
            if random.random() < 0.7:
                # Python Pi with random iterations
                iterations = random.randint(500, 1500)
                req = FunctionRequest('python-pi', size=iterations)
                yield env.process(env.faas.invoke(req))
            else:
                # ResNet50 with small images
                yield env.process(self._simulate_image_transfer(env, 
                    random.choice(edge_nodes), 
                    random.choice(cloud_nodes), 
                    size_mb=random.uniform(0.5, 2)))
                    
                # Small image = size 1.0
                req = FunctionRequest('resnet50-inference', size=1.0)
                yield env.process(env.faas.invoke(req))
                
        logger.info("LIGHT LOAD phase completed")
    
    def _run_medium_load_phase(self, env, edge_nodes, cloud_nodes):
        """Simulate medium load (steady stream of requests)"""
        logger.info("Starting MEDIUM LOAD phase (120 seconds)")
        phase_end = env.now + 120  # 2 minute phase
        
        while env.now < phase_end:
            # Generate requests every 0.5-1.5 seconds
            yield env.timeout(random.uniform(0.5, 1.5))
            
            # Run multiple requests in parallel
            processes = []
            
            # 2-4 parallel requests
            num_requests = random.randint(2, 4)
            
            for _ in range(num_requests):
                # 60% chance of Python Pi, 40% chance of ResNet50
                if random.random() < 0.6:
                    # Python Pi with medium iterations
                    iterations = random.randint(1000, 3000)
                    req = FunctionRequest('python-pi', size=iterations)
                    processes.append(env.process(env.faas.invoke(req)))
                else:
                    # ResNet50 with medium images
                    yield env.process(self._simulate_image_transfer(env, 
                        random.choice(edge_nodes), 
                        random.choice(cloud_nodes), 
                        size_mb=random.uniform(2, 5)))
                        
                    # Medium image with small batch = size 2.0 + batch_size*10
                    batch_size = random.randint(1, 2)
                    req = FunctionRequest('resnet50-inference', size=2.0 + batch_size*10)
                    processes.append(env.process(env.faas.invoke(req)))
            
            # Wait for some of the processes to complete (but not all, to simulate overlapping requests)
            if processes:
                yield processes[0]
                
        logger.info("MEDIUM LOAD phase completed")
    
    def _run_heavy_load_phase(self, env, edge_nodes, cloud_nodes):
        """Simulate heavy load (high request rate)"""
        logger.info("Starting HEAVY LOAD phase (90 seconds)")
        phase_end = env.now + 90  # 1.5 minute phase
        
        while env.now < phase_end:
            # Generate batches of requests every 0.2-0.8 seconds
            yield env.timeout(random.uniform(0.2, 0.8))
            
            # Run multiple requests in parallel
            processes = []
            
            # 5-10 parallel requests
            num_requests = random.randint(5, 10)
            
            for _ in range(num_requests):
                # 50% chance of Python Pi, 50% chance of ResNet50
                if random.random() < 0.5:
                    # Python Pi with high iterations
                    iterations = random.randint(2000, 5000)
                    req = FunctionRequest('python-pi', size=iterations)
                    processes.append(env.process(env.faas.invoke(req)))
                else:
                    # ResNet50 with large images and bigger batches
                    yield env.process(self._simulate_image_transfer(env, 
                        random.choice(edge_nodes), 
                        random.choice(cloud_nodes), 
                        size_mb=random.uniform(5, 15)))
                        
                    # Image size (1=small, 2=medium, 4=large) + batch_size*10
                    image_size = random.choice([2.0, 4.0])  # medium or large
                    batch_size = random.randint(1, 4)
                    req = FunctionRequest('resnet50-inference', size=image_size + batch_size*10)
                    processes.append(env.process(env.faas.invoke(req)))
            
            # Wait for some of the processes to complete
            if len(processes) > 3:
                yield processes[0]
                yield processes[1]
                
        logger.info("HEAVY LOAD phase completed")
    
    def _run_bursty_load_phase(self, env, edge_nodes, cloud_nodes):
        """Simulate bursty traffic with periods of very high load and then quiet"""
        logger.info("Starting BURSTY LOAD phase (180 seconds)")
        phase_end = env.now + 180  # 3 minute phase
        
        while env.now < phase_end:
            # 20% chance of burst, 80% chance of quiet period
            if random.random() < 0.2:
                # Burst period: 5-15 seconds of high activity
                logger.info("Starting traffic burst")
                burst_end = env.now + random.uniform(5, 15)
                
                while env.now < burst_end and env.now < phase_end:
                    # Generate many requests very quickly
                    yield env.timeout(random.uniform(0.05, 0.2))
                    
                    # 10-20 parallel requests in burst
                    processes = []
                    num_requests = random.randint(10, 20)
                    
                    # Also simulate large data transfers
                    for edge_node in random.sample(edge_nodes, min(3, len(edge_nodes))):
                        for cloud_node in random.sample(cloud_nodes, min(2, len(cloud_nodes))):
                            yield env.process(self._simulate_image_transfer(env, 
                                edge_node, cloud_node, 
                                size_mb=random.uniform(10, 50)))
                    
                    for _ in range(num_requests):
                        # Mix of Python Pi and ResNet50 with large parameters
                        if random.random() < 0.4:
                            iterations = random.randint(3000, 8000)
                            req = FunctionRequest('python-pi', size=iterations)
                            processes.append(env.process(env.faas.invoke(req)))
                        else:
                            # Large image (4.0) + large batch size (20-80)
                            batch_size = random.randint(2, 8)
                            req = FunctionRequest('resnet50-inference', size=4.0 + batch_size*10)
                            processes.append(env.process(env.faas.invoke(req)))
                
                logger.info("Traffic burst ended")
            else:
                # Quiet period: 10-30 seconds of low activity
                quiet_end = env.now + random.uniform(10, 30)
                
                while env.now < quiet_end and env.now < phase_end:
                    # Generate very few requests
                    yield env.timeout(random.uniform(3, 8))
                    
                    # Only 1-2 requests
                    if random.random() < 0.7:
                        iterations = random.randint(500, 1000)
                        req = FunctionRequest('python-pi', size=iterations)
                        yield env.process(env.faas.invoke(req))
                    else:
                        yield env.process(self._simulate_image_transfer(env, 
                            random.choice(edge_nodes), 
                            random.choice(cloud_nodes), 
                            size_mb=random.uniform(0.5, 2)))
                            
                        # Small image (1.0) with batch size 1
                        req = FunctionRequest('resnet50-inference', size=1.0)
                        yield env.process(env.faas.invoke(req))
                
        logger.info("BURSTY LOAD phase completed")

    def _simulate_image_transfer(self, env, source, destination, size_mb):
        """Simulate network traffic for image transfer"""
        logger.info(f"Transferring {size_mb:.1f}MB from {source} to {destination}")
        try:
            # Convert MB to bytes
            size_bytes = size_mb * 1024 * 1024
            start_time = env.now
            
            # Check if appropriate network functions are available
            if hasattr(env.topology, 'route_by_node_name'):
                # Get route between nodes
                route = env.topology.route_by_node_name(source, destination)
                
                # Check if the environment has SafeFlow for proper network simulation
                if 'sim.net' in sys.modules and hasattr(sys.modules['sim.net'], 'SafeFlow'):
                    from sim.net import SafeFlow
                    flow = SafeFlow(env, size_bytes, route)
                    yield flow.start()
                    
                    # Log network usage per hop
                    if hasattr(env, 'metrics'):
                        for hop in route.hops:
                            env.metrics.log_network(size_bytes, 'data_transfer', hop)
                        env.metrics.log_flow(size_bytes, env.now - start_time, source, destination, 'data_transfer')
                else:
                    # No SafeFlow available, use simple timeout
                    bandwidth = getattr(route, 'bandwidth', 100 * 1024 * 1024)  # Default 100 Mbps
                    transfer_time = size_bytes / bandwidth
                    yield env.timeout(transfer_time)
            else:
                # No routing available, use simple timeout
                bandwidth = 100 * 1024 * 1024  # 100 Mbps in bytes per second
                transfer_time = size_bytes / bandwidth
                logger.info(f"Simulating transfer time of {transfer_time:.2f} seconds")
                yield env.timeout(transfer_time)
                
        except Exception as e:
            logger.error(f"Error transferring data: {e}")
            yield env.timeout(0)  # Return a proper generator

    def prepare_deployments(self) -> List[FunctionDeployment]:
        """Prepare all function deployments with realistic configurations"""
        python_pi_fd = self.prepare_python_pi_deployment()
        resnet_fd = self.prepare_resnet_inference_deployment()
        
        return [python_pi_fd, resnet_fd]

    def prepare_python_pi_deployment(self):
        """Prepare Python Pi function deployment with realistic scaling"""
        # Design Time
        python_pi = 'python-pi'
        python_pi_cpu = FunctionImage(image='python-pi-cpu')
        python_pi_fn = Function(python_pi, fn_images=[python_pi_cpu])

        # Run time
        python_pi_fn_container = FunctionContainer(
            python_pi_cpu,
            resource_config=KubernetesResourceConfiguration.create_from_str(
                cpu='250m', memory='128Mi')
        )

        # Create and configure scaling for more realistic behavior
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 2  # Start with 2 replicas
        scaling_config.scale_max = 10  # Allow scaling up to 10
        scaling_config.target_average_utilization = 0.7  # Scale at 70% CPU
        
        python_pi_fd = FunctionDeployment(
            python_pi_fn,
            [python_pi_fn_container],
            scaling_config
        )

        return python_pi_fd

    def prepare_resnet_inference_deployment(self):
        """Prepare ResNet50 function deployment with GPU preference"""
        # Design time
        resnet_inference = 'resnet50-inference'
        inference_cpu = 'resnet50-inference-cpu'
        inference_gpu = 'resnet50-inference-gpu'

        resnet_inference_gpu = FunctionImage(image=inference_gpu)
        resnet_inference_cpu = FunctionImage(image=inference_cpu)
        resnet_fn = Function(resnet_inference, fn_images=[resnet_inference_gpu, resnet_inference_cpu])

        # Run time
        # CPU version (higher CPU request)
        resnet_cpu_container = FunctionContainer(
            resnet_inference_cpu,
            resource_config=KubernetesResourceConfiguration.create_from_str(
                cpu='1000m', memory='2048Mi')
        )

        # GPU version (lower CPU but more memory)
        resnet_gpu_container = FunctionContainer(
            resnet_inference_gpu,
            resource_config=KubernetesResourceConfiguration.create_from_str(
                cpu='500m', memory='3072Mi')
        )

        # Create and configure scaling
        scaling_config = ScalingConfiguration()
        scaling_config.scale_min = 1  # Start with 1 replica
        scaling_config.scale_max = 8  # Scale up to 8 replicas
        scaling_config.target_average_utilization = 0.6  # Scale at 60% CPU
        
        resnet_fd = FunctionDeployment(
            resnet_fn,
            [resnet_gpu_container, resnet_cpu_container],
            scaling_config,
            DeploymentRanking([inference_gpu, inference_cpu])
        )

        return resnet_fd


if __name__ == '__main__':
    main()
