def calculate_energy(
    device_type, cpu_util, disk_io_kb_s, gpu_util, memory_kb, network_mb_s, duration_s
):
    """
    Calculate energy consumption in Joules based on resource utilization

    Args:
        device_type: Type of device (e.g., 'nx', 'tx2', 'rpi4')
        cpu_util: CPU utilization as a fraction (0.0-1.0)
        disk_io_kb_s: Disk I/O rate in KB/s
        gpu_util: GPU utilization as a fraction (0.0-1.0)
        memory_kb: Memory usage in KB
        network_mb_s: Network I/O rate in MB/s
        duration_s: Duration in seconds

    Returns:
        Energy consumption in Joules
    """
    # Device-specific power coefficients (in Watts)
    if device_type == "nx":  # Jetson Xavier NX
        base_power = 5.0  # Idle power consumption
        cpu_coeff = 10.0  # Additional power at 100% CPU
        gpu_coeff = 15.0  # Additional power at 100% GPU
        memory_coeff = 0.05 / 1024  # W per MB of active memory
        disk_coeff = 0.02 / 1024  # W per KB/s of disk I/O
        network_coeff = 0.1  # W per MB/s of network traffic
    elif device_type == "tx2":  # Jetson TX2
        base_power = 3.0
        cpu_coeff = 7.5
        gpu_coeff = 12.0
        memory_coeff = 0.04 / 1024
        disk_coeff = 0.015 / 1024
        network_coeff = 0.08
    elif device_type == "rpi4":  # Raspberry Pi 4
        base_power = 2.0
        cpu_coeff = 3.5
        gpu_coeff = 0.5  # Limited GPU capabilities
        memory_coeff = 0.025 / 1024
        disk_coeff = 0.01 / 1024
        network_coeff = 0.05
    elif device_type == "rpi3":  # Raspberry Pi 3
        base_power = 1.5
        cpu_coeff = 2.5
        gpu_coeff = 0.2
        memory_coeff = 0.02 / 1024
        disk_coeff = 0.008 / 1024
        network_coeff = 0.04
    elif device_type.startswith("coral"):  # Coral devices with TPU
        base_power = 2.5
        cpu_coeff = 3.0
        gpu_coeff = 0.0  # Uses TPU instead
        tpu_coeff = 4.0  # TPU power consumption
        memory_coeff = 0.03 / 1024
        disk_coeff = 0.01 / 1024
        network_coeff = 0.05
    else:  # Default x86 server
        base_power = 80.0  # Higher idle power
        cpu_coeff = 120.0  # Significant power increase with CPU load
        gpu_coeff = 150.0  # High-end GPU power consumption
        memory_coeff = 0.1 / 1024
        disk_coeff = 0.05 / 1024
        network_coeff = 0.2

    # Calculate power consumption components
    power_base = base_power
    power_cpu = cpu_util * cpu_coeff
    power_gpu = gpu_util * gpu_coeff
    power_memory = memory_kb * memory_coeff
    power_disk = disk_io_kb_s * disk_coeff
    power_network = network_mb_s * network_coeff

    # Special case for Coral TPU
    if device_type.startswith("coral"):
        power_gpu = gpu_util * tpu_coeff  # Use TPU coefficient instead

    # Total power consumption
    total_power = (
        power_base + power_cpu + power_gpu + power_memory + power_disk + power_network
    )

    # Energy = Power × Time
    energy_joules = total_power * duration_s

    return {
        "energy_joules": energy_joules,
        "energy_wh": energy_joules / 3600,
        "avg_power": total_power,
        "components": {
            "base": power_base,
            "cpu": power_cpu,
            "gpu": power_gpu,
            "memory": power_memory,
            "disk": power_disk,
            "network": power_network,
        },
    }
