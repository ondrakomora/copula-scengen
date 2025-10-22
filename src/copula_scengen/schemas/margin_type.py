from enum import Enum


class MarginType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
