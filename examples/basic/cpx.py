import logging
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from sim.core import Environment
from sim.faas import FunctionDeployment, FunctionContainer, ScalingConfiguration
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation
from sim.benchmark import Benchmark
from sim.metrics import Metrics, RuntimeLogger
from sim.docker import ContainerRegistry, ImageProperties
from sim.skippy import SimulationClusterContext
from sim.topology import Topology
from skippy.core.scheduler import Scheduler

logger = logging.getLogger(__name__)


class BaseBenchmark(Benchmark):  # Changed to inherit from Benchmark
    """A benchmark that simulates realistic workload patterns with multiple functions"""

    def __init__(self, duration=300):
        self.duration = duration
        self.functions = [
            "image-recognition",  # Compute-intensive, GPU acceleration
            "data-processing",  # Memory-intensive
            "api-gateway",  # Low resource, high request count
            "video-analytics",  # Both compute and memory intensive
        ]


def example_topology() -> Topology:
    """Create a basic example topology with a mix of different node types"""
    topology = Topology()

    # Add cloud servers (high capacity)
    for i in range(3):
        # Try different ways of adding nodes
        try:
            # Method 1: Use the signature topology.add_node(name, props)
            topology.add_node(
                f"server_{i}",
                {
                    "type": "cloud_server",
                    "region": "us-east",
                    "cpu_cores": 32,
                    "memory_mb": 131072,  # 128 GB
                    "gpu": i < 2,  # 2 servers with GPU
                    "arch": "x86_64",
                },
            )
        except TypeError:
            try:
                # Method 2: Pass properties as separate arguments
                topology.add_node(
                    f"server_{i}",
                    type="cloud_server",
                    region="us-east",
                    cpu_cores=32,
                    memory_mb=131072,
                    gpu=i < 2,
                    arch="x86_64",
                )
            except TypeError:
                # Method 3: Try with just the node name
                topology.add_node(f"server_{i}")
                # Then try to set properties another way if needed
                # This would depend on your specific Topology implementation

    # Add edge servers (medium capacity) - using Method 2
    for i in range(5):
        try:
            topology.add_node(
                f"edge_{i}",
                type="edge_server",
                region=f"region_{i//2}",
                cpu_cores=16,
                memory_mb=32768,
                gpu=(i % 2 == 0),
                arch="x86_64",
            )
        except TypeError:
            # Simplified version with just the name if Method 2 fails
            topology.add_node(f"edge_{i}")

    # Add Raspberry Pi devices (low capacity) - using Method 3
    for i in range(10):
        topology.add_node(f"rpi_{i}")

    # Connect cloud servers to each other (low latency)
    for i in range(3):
        for j in range(i + 1, 3):
            try:
                topology.add_edge(f"server_{i}", f"server_{j}", latency=2.0)
            except TypeError:
                # Try alternative edge creation if the above fails
                try:
                    topology.add_edge(f"server_{i}", f"server_{j}", {"latency": 2.0})
                except TypeError:
                    # Simplest version
                    topology.add_edge(f"server_{i}", f"server_{j}")

    # Connect edge servers to cloud servers - simplified
    for i in range(5):
        for j in range(3):
            topology.add_edge(f"edge_{i}", f"server_{j}")

    # Connect Raspberry Pi devices to nearest edge server - simplified
    for i in range(10):
        edge_index = i // 2
        if edge_index >= 5:
            edge_index = 4  # Cap at max edge server

        topology.add_edge(f"rpi_{i}", f"edge_{edge_index}")

    # Create some direct connections between edge servers - simplified
    for i in range(5):
        for j in range(i + 1, 5):
            if abs(i - j) <= 2:  # Only connect nearby edge servers
                topology.add_edge(f"edge_{i}", f"edge_{j}")

    return topology


# Add ExampleBenchmark class to your cpx.py file
class ExampleBenchmark(Benchmark):
    """A simple benchmark that invokes two functions several times"""

    def __init__(self, duration=10):
        self.duration = duration

    def setup(self, env):
        """Set up the benchmark by registering images and deploying functions"""
        # Register container images
        if hasattr(env, "container_registry"):
            logger.info("Registering container images...")

            # Image for ML inference
            env.container_registry.register_image(
                ImageProperties("resnet50-inference-cpu", 56000000, "latest", "arm32"),
                ImageProperties("resnet50-inference-cpu", 56000000, "latest", "x86"),
                ImageProperties(
                    "resnet50-inference-cpu", 56000000, "latest", "aarch64"
                ),
            )

            env.container_registry.register_image(
                ImageProperties("resnet50-inference-gpu", 56000000, "latest", "arm32"),
                ImageProperties("resnet50-inference-gpu", 56000000, "latest", "x86"),
                ImageProperties(
                    "resnet50-inference-gpu", 56000000, "latest", "aarch64"
                ),
            )

            # Image for compute function
            env.container_registry.register_image(
                ImageProperties("python-pi-cpu", 58000000, "latest", "arm32"),
                ImageProperties("python-pi-cpu", 58000000, "latest", "x86"),
                ImageProperties("python-pi-cpu", 58000000, "latest", "aarch64"),
            )

        # Deploy functions if FaaS system is available
        if hasattr(env, "faas"):
            logger.info("Deploying functions...")

            # ML inference function (prefers GPU)
            resnet_container = FunctionContainer(
                image="resnet50-inference-gpu:latest",
                labels={
                    "cpu": "100m",  # 0.1 CPU core
                    "memory": "1024Mi",  # 1 GB RAM
                    "gpu": "preferred",  # Use GPU if available
                    "workers": "1",
                },
            )

            resnet_deployment = FunctionDeployment(
                name="resnet50-inference",
                fn_containers=[resnet_container],
                scaling_config=ScalingConfiguration(
                    scale_min=1, scale_max=5, target_concurrency=10
                ),
            )

            # CPU-bound compute function
            compute_container = FunctionContainer(
                image="python-pi-cpu:latest",
                labels={
                    "cpu": "1000m",  # 1 CPU core
                    "memory": "512Mi",  # 512 MB RAM
                    "workers": "2",
                },
            )

            compute_deployment = FunctionDeployment(
                name="python-pi",
                fn_containers=[compute_container],
                scaling_config=ScalingConfiguration(
                    scale_min=1, scale_max=5, target_concurrency=5
                ),
            )

            # Deploy the functions
            env.faas.deploy(resnet_deployment)
            env.faas.deploy(compute_deployment)

    def run(self, env):
        """Generate workload by invoking functions"""
        # Wait for setup to complete and functions to be ready
        yield env.timeout(10)

        # Invoke inference function
        logger.info("Executing 10 resnet50-inference requests")
        for i in range(10):
            env.faas.invoke("resnet50-inference", request_id=f"infer-{i}")
            yield env.timeout(0.1)

        # Invoke compute function
        logger.info("Executing 10 python-pi requests")
        for i in range(10):
            env.faas.invoke("python-pi", request_id=f"compute-{i}")
            yield env.timeout(0.1)

        # Run for the specified duration
        remaining = self.duration - env.now
        if remaining > 0:
            yield env.timeout(remaining)


# 1. Define a more complex topology with diverse node types
def create_complex_topology():
    topology = Topology()

    # Cloud nodes - high capacity but high latency
    for i in range(3):
        topology.add_node(
            f"cloud_{i}",
            {
                "type": "cloud",
                "region": "us-east",
                "cpu_cores": 32,
                "memory_mb": 131072,  # 128 GB
                "gpu": True if i < 2 else False,  # 2 with GPU, 1 without
                "power_factor": 2.0,  # Cloud servers consume more power
            },
        )

    # Edge data center nodes - medium capacity, medium latency
    for i in range(5):
        topology.add_node(
            f"edge_dc_{i}",
            {
                "type": "edge_datacenter",
                "region": f"edge_region_{i//2}",
                "cpu_cores": 16,
                "memory_mb": 32768,  # 32 GB
                "gpu": i % 2 == 0,  # Alternating GPU availability
                "power_factor": 1.5,
            },
        )

    # Edge nodes - lower capacity, lower latency
    for i in range(10):
        topology.add_node(
            f"edge_{i}",
            {
                "type": "edge",
                "region": f"edge_region_{i//3}",
                "cpu_cores": 8,
                "memory_mb": 8192,  # 8 GB
                "gpu": i % 4 == 0,  # Every 4th node has a GPU
                "power_factor": 1.0,
            },
        )

    # IoT gateway nodes - very limited but closest to end-users
    for i in range(15):
        topology.add_node(
            f"iot_gw_{i}",
            {
                "type": "iot_gateway",
                "region": f"edge_region_{i//4}",
                "cpu_cores": 4,
                "memory_mb": 2048,  # 2 GB
                "gpu": False,
                "power_factor": 0.5,
            },
        )

    # Define network connections with realistic latencies

    # Cloud to edge data centers (40-60ms)
    for c in range(3):
        for e in range(5):
            latency = random.uniform(40, 60)
            topology.add_edge(f"cloud_{c}", f"edge_dc_{e}", latency=latency)

    # Edge data centers to edge nodes (10-25ms)
    for e_dc in range(5):
        for e in range(10):
            if e // 3 == e_dc // 2:  # Connect to nodes in same region
                latency = random.uniform(10, 25)
                topology.add_edge(f"edge_dc_{e_dc}", f"edge_{e}", latency=latency)

    # Edge nodes to IoT gateways (5-15ms)
    for e in range(10):
        for i in range(15):
            if i // 4 == e // 3:  # Connect to gateways in same region
                latency = random.uniform(5, 15)
                topology.add_edge(f"edge_{e}", f"iot_gw_{i}", latency=latency)

    # Add some direct connections between regions for redundancy
    for i in range(5):
        for j in range(i + 1, 5):
            src = f"edge_dc_{i}"
            dst = f"edge_dc_{j}"
            latency = random.uniform(30, 50)
            topology.add_edge(src, dst, latency=latency)

    return topology


# 2. Define a more realistic workload pattern
class ComplexWorkloadBenchmark(Benchmark):
    """A benchmark that simulates realistic workload patterns with multiple functions"""

    def __init__(self, duration=300):
        self.duration = duration
        self.functions = [
            "image-recognition",  # Compute-intensive, GPU acceleration
            "data-processing",  # Memory-intensive
            "api-gateway",  # Low resource, high request count
            "video-analytics",  # Both compute and memory intensive
        ]

    def run(self, env):
        """Generate a mix of periodic, bursty, and random request patterns"""

        # Track when the next request should be generated for each pattern
        next_periodic = 0
        next_burst = random.uniform(30, 60)
        next_burst_end = next_burst + random.uniform(10, 20)

        # Create request at a fixed rate (periodic pattern)
        while env.now < self.duration:
            # 1. Regular periodic requests for api-gateway (1 req/sec)
            if env.now >= next_periodic:
                self._create_request(env, "api-gateway")
                next_periodic = env.now + 1.0

            # 2. Bursty workload for image-recognition (10 req/sec during burst)
            if next_burst <= env.now < next_burst_end:
                self._create_request(env, "image-recognition")

                # Schedule next burst request
                if env.now + 0.1 < next_burst_end:
                    yield env.timeout(0.1)
                else:
                    # End of burst, schedule next burst
                    next_burst = env.now + random.uniform(20, 40)
                    next_burst_end = next_burst + random.uniform(10, 30)
                    yield env.timeout(random.uniform(1, 3))

            # 3. Random requests for data-processing (avg 1 every 5 sec)
            if random.random() < 0.2 / 10:  # 0.02 probability per 0.1 sec
                self._create_request(env, "data-processing")

            # 4. Periodic video-analytics requests (1 every 10 sec)
            if env.now % 10 < 0.1 and "video-analytics" in self.functions:
                self._create_request(env, "video-analytics")

            # Advance simulation time
            yield env.timeout(0.1)

    def _create_request(self, env, function):
        # Generate a unique request ID
        request_id = f"{function}-{env.now:.2f}"

        # Invoke the function
        env.faas.invoke(function, request_id=request_id)

        # Log the request generation
        logger.debug(f"Generated request {request_id} at t={env.now:.2f}")


# 3. Define a main function to run the simulation
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create topology
    logger.info("Creating complex topology...")
    topology = create_complex_topology()
    logger.info(
        f"Created topology with {len(topology.nodes)} nodes and {len(topology.edges)} connections"
    )

    # Create benchmark
    benchmark = ComplexWorkloadBenchmark(duration=300)  # 5 minute simulation

    # Initialize environment
    env = Environment()
    env.metrics = Metrics(env, RuntimeLogger())
    env.topology = topology
    env.faas = DefaultFaasSystem(env, scale_by_average_requests=True, scale_interval=30)
    env.container_registry = ContainerRegistry()

    # Register container images
    logger.info("Registering container images...")
    env.container_registry.register_image(
        ImageProperties("image-recognition", 175000000, "latest", "x86"),
        ImageProperties("image-recognition", 168000000, "latest", "arm64"),
    )
    env.container_registry.register_image(
        ImageProperties("data-processing", 120000000, "latest", "x86"),
        ImageProperties("data-processing", 115000000, "latest", "arm64"),
    )
    env.container_registry.register_image(
        ImageProperties("api-gateway", 45000000, "latest", "x86"),
        ImageProperties("api-gateway", 42000000, "latest", "arm64"),
    )
    env.container_registry.register_image(
        ImageProperties("video-analytics", 210000000, "latest", "x86"),
        ImageProperties("video-analytics", 205000000, "latest", "arm64"),
    )

    # Set up cluster and scheduler
    env.cluster = SimulationClusterContext(env)
    env.scheduler = Scheduler(env.cluster)

    # Create function deployments
    logger.info("Creating function deployments...")

    # 1. Image Recognition - GPU accelerated when possible
    image_recog_container = FunctionContainer(
        image="image-recognition:latest",
        labels={
            "cpu": "2000m",  # 2 CPU cores
            "memory": "4096Mi",  # 4 GB RAM
            "gpu": "preferred",  # Use GPU if available
            "workers": "1",
        },
    )
    image_recog_deployment = FunctionDeployment(
        name="image-recognition",
        fn_containers=[image_recog_container],
        scaling_config=ScalingConfiguration(
            scale_min=2, scale_max=10, target_concurrency=5
        ),
    )

    # 2. Data Processing - Memory intensive
    data_proc_container = FunctionContainer(
        image="data-processing:latest",
        labels={
            "cpu": "1000m",  # 1 CPU core
            "memory": "8192Mi",  # 8 GB RAM
            "workers": "2",
        },
    )
    data_proc_deployment = FunctionDeployment(
        name="data-processing",
        fn_containers=[data_proc_container],
        scaling_config=ScalingConfiguration(
            scale_min=1, scale_max=5, target_concurrency=3
        ),
    )

    # 3. API Gateway - Lightweight
    api_gateway_container = FunctionContainer(
        image="api-gateway:latest",
        labels={
            "cpu": "500m",  # 0.5 CPU core
            "memory": "512Mi",  # 512 MB RAM
            "workers": "4",
        },
    )
    api_gateway_deployment = FunctionDeployment(
        name="api-gateway",
        fn_containers=[api_gateway_container],
        scaling_config=ScalingConfiguration(
            scale_min=4, scale_max=12, target_concurrency=10
        ),
    )

    # 4. Video Analytics - Compute and memory intensive
    video_analytics_container = FunctionContainer(
        image="video-analytics:latest",
        labels={
            "cpu": "4000m",  # 4 CPU cores
            "memory": "6144Mi",  # 6 GB RAM
            "gpu": "required",  # Must have GPU
            "workers": "1",
        },
    )
    video_analytics_deployment = FunctionDeployment(
        name="video-analytics",
        fn_containers=[video_analytics_container],
        scaling_config=ScalingConfiguration(
            scale_min=1, scale_max=3, target_concurrency=2
        ),
    )

    # Deploy functions
    logger.info("Deploying functions...")
    env.faas.deploy(image_recog_deployment)
    env.faas.deploy(data_proc_deployment)
    env.faas.deploy(api_gateway_deployment)
    env.faas.deploy(video_analytics_deployment)

    # Run simulation
    logger.info(f"Starting simulation for {benchmark.duration} seconds...")
    sim = Simulation(topology, benchmark, env=env)
    sim.run()
    logger.info("Simulation completed")

    # Analyze results
    analyze_results(env)


def analyze_results(env):
    """Analyze and visualize simulation results"""
    # Ensure output directory exists
    output_dir = "complex_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Analyzing results and saving to {output_dir}...")

    # Extract metric DataFrames
    dfs = {
        "invocations_df": env.metrics.extract_dataframe("invocations"),
        "scale_df": env.metrics.extract_dataframe("scale"),
        "schedule_df": env.metrics.extract_dataframe("schedule"),
        "replica_deployment_df": env.metrics.extract_dataframe("replica_deployment"),
        "network_df": env.metrics.extract_dataframe("network"),
        "node_utilization_df": env.metrics.extract_dataframe("node_utilization"),
    }

    # Save DataFrames to CSV
    for name, df in dfs.items():
        if df is not None and not df.empty:
            df.to_csv(f"{output_dir}/{name}.csv", index=False)

    # Print basic statistics
    print("\n=== SIMULATION RESULTS ===")

    # Invocation statistics
    if "invocations_df" in dfs and dfs["invocations_df"] is not None:
        inv_df = dfs["invocations_df"]
        total_invocations = len(inv_df)
        print(f"\nTotal invocations: {total_invocations}")

        # Group by function
        func_stats = inv_df.groupby("function").agg(
            {"duration": ["count", "mean", "min", "max"], "wait_time": ["mean", "max"]}
        )
        print("\nFunction Statistics:")
        print(func_stats)

        # Success rate
        if "success" in inv_df.columns:
            success_rate = inv_df["success"].mean() * 100
            print(f"\nOverall success rate: {success_rate:.2f}%")

        # Create visualizations

        # 1. Execution time by function
        plt.figure(figsize=(10, 6))
        for func in inv_df["function"].unique():
            data = inv_df[inv_df["function"] == func]["duration"]
            plt.hist(data, alpha=0.5, bins=20, label=func)
        plt.title("Function Execution Time Distribution")
        plt.xlabel("Execution Time (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"{output_dir}/execution_times.png")

        # 2. Request timeline
        plt.figure(figsize=(14, 8))
        for func in inv_df["function"].unique():
            data = inv_df[inv_df["function"] == func]
            plt.scatter(data["t_start"], data["duration"], alpha=0.5, label=func, s=10)
        plt.title("Function Invocation Timeline")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Execution Duration (s)")
        plt.legend()
        plt.savefig(f"{output_dir}/invocation_timeline.png")

        # 3. Node distribution
        if "node" in inv_df.columns:
            plt.figure(figsize=(12, 6))
            node_counts = inv_df.groupby(["function", "node"]).size().unstack()
            node_counts.plot(kind="bar", stacked=True)
            plt.title("Function Invocations by Node")
            plt.xlabel("Function")
            plt.ylabel("Number of Invocations")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/node_distribution.png")

    # Node utilization
    if "node_utilization_df" in dfs and dfs["node_utilization_df"] is not None:
        node_df = dfs["node_utilization_df"]
        if not node_df.empty:
            plt.figure(figsize=(14, 8))

            # Group by node and compute average
            node_util = node_df.groupby(["node", "t"]).mean().reset_index()

            # Plot time series for each node
            for node in node_util["node"].unique():
                node_data = node_util[node_util["node"] == node]
                if "cpu_util" in node_data.columns:
                    plt.plot(node_data["t"], node_data["cpu_util"], "-", label=node)

            plt.title("CPU Utilization Over Time")
            plt.xlabel("Simulation Time (s)")
            plt.ylabel("CPU Utilization (%)")
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/cpu_utilization.png")

    # Scaling activity
    if "scale_df" in dfs and dfs["scale_df"] is not None:
        scale_df = dfs["scale_df"]
        if not scale_df.empty:
            plt.figure(figsize=(12, 6))

            pivot_scale = scale_df.pivot_table(
                index="t", columns="function", values="replicas", aggfunc="mean"
            ).fillna(method="ffill")

            pivot_scale.plot(marker="o", markersize=4)
            plt.title("Function Scaling Over Time")
            plt.xlabel("Simulation Time (s)")
            plt.ylabel("Number of Replicas")
            plt.grid(True)
            plt.savefig(f"{output_dir}/scaling.png")

    # Create summary report
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write("COMPLEX FAAS SIMULATION SUMMARY\n")
        f.write("==============================\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if "invocations_df" in dfs and dfs["invocations_df"] is not None:
            inv_df = dfs["invocations_df"]
            f.write(f"Total invocations: {len(inv_df)}\n")
            f.write(
                f"Simulation duration: {inv_df['t_start'].max() - inv_df['t_start'].min():.2f} s\n\n"
            )

            f.write("Function Statistics:\n")
            for func in inv_df["function"].unique():
                func_data = inv_df[inv_df["function"] == func]
                f.write(f"  {func}:\n")
                f.write(f"    Invocations: {len(func_data)}\n")
                f.write(
                    f"    Avg execution time: {func_data['duration'].mean():.4f} s\n"
                )
                if "wait_time" in func_data.columns:
                    f.write(
                        f"    Avg wait time: {func_data['wait_time'].mean():.4f} s\n"
                    )
                f.write("\n")

        if "network_df" in dfs and dfs["network_df"] is not None:
            net_df = dfs["network_df"]
            if not net_df.empty and "size" in net_df.columns:
                total_bytes = net_df["size"].sum()
                f.write(f"Total network traffic: {total_bytes/1000000:.2f} MB\n\n")

    logger.info(f"Analysis complete. Results saved to {output_dir}")
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
