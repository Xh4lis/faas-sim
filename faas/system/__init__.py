from .core import FunctionContainer, Function, FunctionImage, DeploymentRanking


# Add any additional imports needed
class ScalingConfiguration:
    def __init__(self, scale_min=1, scale_max=1):
        self.scale_min = scale_min
        self.scale_max = scale_max
