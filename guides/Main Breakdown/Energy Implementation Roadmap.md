# Implementing Energy Modeling in the FaaS Simulation Framework

Here's a high-level approach to implementing energy modeling in your FaaS simulation using the existing resource tracking system:

## Step 1: Create an Energy Model Class

```python
class EnergyModel:
    def __init__(self, base_power=5.0, coefficients=None):
        # Default coefficients if none provided
        self.base_power = base_power  # Idle power in Watts
        self.coefficients = coefficients or {
            'cpu': 10.0,     # W per unit CPU utilization
            'memory': 0.01,  # W per MB
            'gpu': 15.0,     # W per unit GPU utilization
            'blkio': 0.05,   # W per MB/s
            'net': 0.08      # W per MB/s
        }

    def calculate_power(self, resource_utilization):
        """Calculate power consumption based on resource utilization"""
        power = self.base_power

        for resource, coef in self.coefficients.items():
            usage = resource_utilization.get_resource(resource) or 0
            power += usage * coef

        return power
```

## Step 2: Extend ResourceMonitor to Track Energy

Modify the `ResourceMonitor` class to include energy tracking:

```python
class EnergyAwareResourceMonitor(ResourceMonitor):
    def __init__(self, env, reconcile_interval=0.2, logging=True):
        super().__init__(env, reconcile_interval, logging)
        self.energy_models = {}  # Map node names to their energy models
        self.energy_consumption = defaultdict(float)  # Track total energy by node
        self.last_power_reading = defaultdict(float)  # Last power reading by node
        self.last_timestamp = 0

    def add_energy_model(self, node_name, model):
        """Add an energy model for a specific node"""
        self.energy_models[node_name] = model

    def run(self):
        """Override run method to track energy consumption"""
        faas = self.env.faas
        current_time = self.env.now
        self.last_timestamp = current_time

        while True:
            yield self.env.timeout(self.reconcile_interval)

            current_time = self.env.now
            elapsed = current_time - self.last_timestamp

            # Track energy for each node
            for node_name, node_util in self.env.resource_state.node_resource_utilizations.items():
                if node_name in self.energy_models:
                    # Get total resources for node
                    total_util = node_util.total_utilization

                    # Calculate power at this moment
                    power = self.energy_models[node_name].calculate_power(total_util)

                    # Log power as a resource for monitoring
                    self.env.metrics.log('power', {
                        'node': node_name,
                        'timestamp': current_time,
                        'watts': power
                    })

                    # Track cumulative energy (power × time)
                    energy_joules = power * elapsed
                    self.energy_consumption[node_name] += energy_joules

                    # Log cumulative energy
                    self.env.metrics.log('energy', {
                        'node': node_name,
                        'timestamp': current_time,
                        'joules': self.energy_consumption[node_name]
                    })

            self.last_timestamp = current_time
```

## Step 3: Integration with the Environment

Modify your simulation setup to use the energy-aware resource monitor:

```python
def setup_environment_with_energy_tracking(env, node_types):
    # Create energy models for each node type
    energy_models = {
        'nx': EnergyModel(base_power=2.5, coefficients={
            'cpu': 8.5, 'memory': 0.008, 'gpu': 12.0, 'blkio': 0.04, 'net': 0.07
        }),
        'nano': EnergyModel(base_power=1.8, coefficients={
            'cpu': 5.0, 'memory': 0.006, 'gpu': 10.0, 'blkio': 0.03, 'net': 0.05
        }),
        'xeongpu': EnergyModel(base_power=50.0, coefficients={
            'cpu': 25.0, 'memory': 0.012, 'gpu': 75.0, 'blkio': 0.1, 'net': 0.15
        })
    }

    # Create energy-aware resource monitor
    monitor = EnergyAwareResourceMonitor(env)

    # Assign energy models to nodes
    for node in env.topology.get_nodes():
        for node_type, model in energy_models.items():
            if node_type in node.name:
                monitor.add_energy_model(node.name, model)
                break

    # Replace default resource monitor
    env.resource_monitor = monitor
    env.process(monitor.run())

    return env
```

## Step 4: Analysis and Reporting

Add energy metrics extraction to your analysis code:

```python
def extract_energy_metrics(sim):
    # Extract the energy metrics dataframe
    energy_df = sim.env.metrics.extract_dataframe('energy')
    power_df = sim.env.metrics.extract_dataframe('power')

    # Calculate total energy consumption
    total_energy = energy_df.groupby('node')['joules'].max().sum()

    # Calculate energy per function invocation
    invocations_df = sim.env.metrics.extract_dataframe('invocations')
    invocation_count = len(invocations_df)
    energy_per_invocation = total_energy / invocation_count if invocation_count > 0 else 0

    print(f"Total energy consumption: {total_energy:.2f} Joules")
    print(f"Energy per invocation: {energy_per_invocation:.2f} Joules")

    return {
        'energy_df': energy_df,
        'power_df': power_df,
        'total_energy': total_energy,
        'energy_per_invocation': energy_per_invocation
    }
```

This approach leverages the existing resource monitoring infrastructure to add energy modeling. You would calibrate the coefficients based on real measurements from your target hardware or published power specifications.
