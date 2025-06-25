from sim.core import Environment
from sim.faas import FunctionDeployment, FunctionContainer, ScalingConfiguration
from sim.faas.system import DefaultFaasSystem
from sim.faassim import Simulation, SimpleSimulatorFactory
from ext.raith21.benchmark.constant import ConstantBenchmark
from sim.metrics import Metrics, RuntimeLogger
from sim.docker import ContainerRegistry
from sim.skippy import SimulationClusterContext

# Fix the import error by importing Topology from the correct module
from sim.topology import Topology
from skippy.core.scheduler import Scheduler
import argparse
import os

# Import the analyzer module
# Make sure faas_sim_analyzer.py is in the same directory
from scripts.faas_sim_analyzer import analyze_simulation_results


def run_simulation():
    """Run a basic faas-sim simulation and analyze the results"""
    # Set up environment
    env = Environment()
    env.simulator_factory = SimpleSimulatorFactory()
    env.metrics = Metrics(env, RuntimeLogger())
    env.container_registry = ContainerRegistry()
    env.cluster = SimulationClusterContext(env)
    env.scheduler = Scheduler(env.cluster)

    # Create basic topology
    topology = Topology()

    # Add edge nodes (Raspberry Pi)
    for i in range(14):
        topology.add_node(
            f"rpi3_{i}",
            {
                "type": "edge",
                "cpu_cores": 4,
                "memory_mb": 23012864 // 1024,  # Convert to MB
                "arch": "arm32",
            },
        )

    # Add fog nodes (NUC)
    for i in range(3):
        topology.add_node(
            f"nuc_{i}",
            {
                "type": "fog",
                "cpu_cores": 4,
                "memory_mb": 179869184 // 1024,  # Convert to MB
                "arch": "x86",
            },
        )

    # Add edge nodes (Jetson TX2)
    for i in range(14):
        topology.add_node(
            f"tx2_{i}",
            {
                "type": "edge",
                "cpu_cores": 4,
                "memory_mb": 40386048 // 1024,  # Convert to MB
                "arch": "aarch64",
                "gpu": True,
            },
        )

    # Add cloud servers
    for i in range(10):
        topology.add_node(
            f"server_{i}",
            {
                "type": "cloud",
                "cpu_cores": 88,
                "memory_mb": 8000000000 // 1024,  # Convert to MB
                "arch": "x86",
                "gpu": True,
            },
        )

    # Add network connections (simplified)
    for i in range(14):
        topology.add_edge(f"rpi3_{i}", f"server_0", latency=50)  # 50ms latency

    for i in range(3):
        topology.add_edge(f"nuc_{i}", f"server_0", latency=30)  # 30ms latency

    for i in range(14):
        topology.add_edge(f"tx2_{i}", f"server_0", latency=40)  # 40ms latency

    # Connect servers
    for i in range(1, 10):
        topology.add_edge(f"server_0", f"server_{i}", latency=10)  # 10ms latency

    env.topology = topology
    env.faas = DefaultFaasSystem(env)

    # Create function deployments
    python_pi = FunctionContainer(
        image="python-pi-cpu:latest",
        labels={"cpu": "1000m", "memory": "48Mi", "workers": "2"},
    )
    python_pi_deployment = FunctionDeployment(
        name="python-pi",
        fn_containers=[python_pi],
        scaling_config=ScalingConfiguration(scale_min=1, scale_max=3),
    )

    resnet50 = FunctionContainer(
        image="resnet50-inference-gpu:latest",
        labels={"cpu": "100m", "memory": "72Mi", "workers": "2", "gpu": "required"},
    )
    resnet50_deployment = FunctionDeployment(
        name="resnet50-inference",
        fn_containers=[resnet50],
        scaling_config=ScalingConfiguration(scale_min=1, scale_max=3),
    )

    # Deploy functions
    env.faas.deploy(python_pi_deployment)
    env.faas.deploy(resnet50_deployment)

    # Create benchmark (10 requests of each function)
    benchmark = ConstantBenchmark(
        ["python-pi", "resnet50-inference"], duration=2, rps=5
    )

    # Create and run simulation
    sim = Simulation(topology, benchmark, env=env)
    sim.run()

    # Analyze and visualize the results
    print("\nAnalyzing simulation results...")
    result_dfs = analyze_simulation_results(env)

    # Inform user about dashboard
    print("\nTo view interactive dashboard, run: python simulation_dashboard.py")

    return result_dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run faas-sim basic example with analysis"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Launch dashboard after simulation"
    )
    args = parser.parse_args()

    # Run simulation and analysis
    results = run_simulation()

    # Optionally launch dashboard
    if args.dashboard:
        import subprocess

        print("\nLaunching dashboard...")
        subprocess.run(["python", "simulation_dashboard.py"])
