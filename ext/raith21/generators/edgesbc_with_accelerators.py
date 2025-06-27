from ext.raith21.generators.edgesbc import edgesbc_settings
from ext.raith21.device import ArchProperties
from ext.raith21.generator import GeneratorSettings
from ext.raith21.model import *

# Create a modified version of edgesbc that includes some accelerators
edgesbc_with_accelerators_settings = GeneratorSettings(
    arch=edgesbc_settings.arch,  # Keep same architecture distribution
    properties={
        **edgesbc_settings.properties,  # Copy existing properties
        # Modify ARM32 to have some TPUs (like Coral devices)
        Arch.ARM32: ArchProperties(
            arch=Arch.ARM32,
            accelerator={
                Accelerator.NONE: 0.7,  # 70% no accelerator (vs 100% in original)
                Accelerator.TPU: 0.3,   # 30% with TPU (Coral Edge TPU)
                Accelerator.GPU: 0
            },
            # Keep other ARM32 properties the same
            cores=edgesbc_settings.properties[Arch.ARM32].cores,
            location=edgesbc_settings.properties[Arch.ARM32].location,
            connection=edgesbc_settings.properties[Arch.ARM32].connection,
            network=edgesbc_settings.properties[Arch.ARM32].network,
            cpu_mhz=edgesbc_settings.properties[Arch.ARM32].cpu_mhz,
            cpu=edgesbc_settings.properties[Arch.ARM32].cpu,
            ram=edgesbc_settings.properties[Arch.ARM32].ram,
            disk=edgesbc_settings.properties[Arch.ARM32].disk,
            gpu_vram={},
            gpu_model={},
            gpu_mhz={}
        )
    }
)