from ext.raith21.generators.edgesbc import edgesbc_settings
from ext.raith21.generator import GeneratorSettings, Arch
from ext.raith21.device import ArchProperties
from ext.raith21.model import *
import copy

def make_edgeai_settings(num_devices, n_nuc):
    settings = copy.deepcopy(edgesbc_settings)
    settings.arch[Arch.X86] = n_nuc / num_devices
    settings.properties[Arch.X86] = ArchProperties(
        arch=Arch.X86,
        accelerator={Accelerator.NONE: 1.0, Accelerator.GPU: 0.0, Accelerator.TPU: 0.0},
        cores={Bins.LOW: 0.0, Bins.MEDIUM: 1.0, Bins.HIGH: 0.0, Bins.VERY_HIGH: 0.0},
        location={Location.EDGE: 1.0, Location.CLOUD: 0.0, Location.MEC: 0.0, Location.MOBILE: 0.0},
        connection={Connection.ETHERNET: 0.0, Connection.WIFI: 0.0, Connection.MOBILE: 1.0},
        network={Bins.LOW: 1.0, Bins.MEDIUM: 0.0, Bins.HIGH: 0.0, Bins.VERY_HIGH: 0.0},
        cpu_mhz={Bins.LOW: 0.0, Bins.MEDIUM: 1.0, Bins.HIGH: 0.0, Bins.VERY_HIGH: 0.0},
        cpu={CpuModel.I7: 1.0, CpuModel.XEON: 0.0},
        ram={Bins.LOW: 0.0, Bins.MEDIUM: 0.0, Bins.HIGH: 1.0, Bins.VERY_HIGH: 0.0},
        disk={Disk.NVME: 1.0, Disk.SSD: 0.0, Disk.SD: 0.0, Disk.FLASH: 0.0, Disk.HDD: 0.0},
        gpu_vram={},  # No GPU
        gpu_model={},
        gpu_mhz={},
    )
    return settings