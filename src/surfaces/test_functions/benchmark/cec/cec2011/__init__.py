# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC 2011 Real-World Optimization Problems.

This module provides benchmark functions from the CEC 2011 competition
on testing evolutionary algorithms on real-world optimization problems.

Problems P01-P13 are implemented from scratch using mathematical formulations.

Problems
--------
P01 (FMSynthesis) : D=6, bounds=[-6.4, 6.35]
    FM Sound Synthesis Parameter Estimation
P02 (LennardJonesPotential) : D=30, bounds=[-4, 4]
    Lennard-Jones Minimum Energy Cluster (10 atoms)
P03 (BifunctionalCatalyst) : D=1, bounds=[-0.6, 0.9]
    Bifunctional Catalyst Blend Optimization
P04 (StirredTankReactor) : D=1, bounds=[0, 5]
    Stirred Tank Reactor Residence Time
P05 (TersoffPotentialSiB) : D=30, bounds=[-10, 10]
    Tersoff Potential Parameter Fitting (Si-B)
P06 (TersoffPotentialSiC) : D=30, bounds=[-10, 10]
    Tersoff Potential Parameter Fitting (Si-C)
P07 (RadarPolyphaseCode) : D=20, bounds=[0, 2*pi]
    Radar Polyphase Code Design (PSL minimization)
P08 (SpreadSpectrumRadar) : D=7, bounds=[0, 15]
    Spread Spectrum Radar Code Design (Merit Factor)
P09 (CircularAntennaArray) : D=12, bounds=[0.1, 2.0]
    Circular Antenna Array Design (Sidelobe minimization)
P10 (DynamicEconomicDispatch) : D=120, bounds=[0, 300]
    Dynamic Economic Dispatch with Valve-Point Effect
P11 (HydrothermalScheduling) : D=24, bounds=[5, 15]
    Short-Term Hydrothermal Scheduling
P12 (Cassini2) : D=22, bounds=[0, 1] (normalized)
    Cassini 2 Spacecraft Trajectory (E-V-V-E-J-S)
P13 (Messenger) : D=26, bounds=[0, 1] (normalized)
    Messenger Spacecraft Trajectory (E-V-V-M-M-M-M)

References
----------
Das, S. & Suganthan, P. N. (2010). Problem Definitions and Evaluation
Criteria for CEC 2011 Competition on Testing Evolutionary Algorithms
on Real World Optimization Problems. Technical Report.
"""

from ._base_cec2011 import CEC2011Function
from .engineering_problems import (
    # Engineering problems P09-P11
    CircularAntennaArray,
    DynamicEconomicDispatch,
    HydrothermalScheduling,
    CEC2011_ENGINEERING,
)
from .functions import (
    # Simple 1D problems
    BifunctionalCatalyst,
    StirredTankReactor,
    # Signal processing problems
    FMSynthesis,
    RadarPolyphaseCode,
    SpreadSpectrumRadar,
    # Molecular/potential problems
    LennardJonesPotential,
    TersoffPotentialSiB,
    TersoffPotentialSiC,
    # Collections from functions.py
    CEC2011_MOLECULAR,
    CEC2011_SIGNAL,
    CEC2011_SIMPLE,
)
from .spacecraft_problems import (
    # Spacecraft trajectory problems P12-P22
    Cassini2,
    Messenger,
    Cassini1,
    GTOC1,
    Rosetta,
    Sagas,
    Cassini2Tight,
    MessengerTight,
    GTOC1Tight,
    RosettaTight,
    SagasTight,
    CEC2011_SPACECRAFT,
)

# Complete collection of all CEC 2011 functions (P01-P22)
CEC2011_ALL = [
    FMSynthesis,  # P01
    LennardJonesPotential,  # P02
    BifunctionalCatalyst,  # P03
    StirredTankReactor,  # P04
    TersoffPotentialSiB,  # P05
    TersoffPotentialSiC,  # P06
    RadarPolyphaseCode,  # P07
    SpreadSpectrumRadar,  # P08
    CircularAntennaArray,  # P09
    DynamicEconomicDispatch,  # P10
    HydrothermalScheduling,  # P11
    Cassini2,  # P12
    Messenger,  # P13
    Cassini1,  # P14
    GTOC1,  # P15
    Rosetta,  # P16
    Sagas,  # P17
    Cassini2Tight,  # P18
    MessengerTight,  # P19
    GTOC1Tight,  # P20
    RosettaTight,  # P21
    SagasTight,  # P22
]

__all__ = [
    # Base class
    "CEC2011Function",
    # Simple problems (P03-P04)
    "BifunctionalCatalyst",
    "StirredTankReactor",
    # Signal processing (P01, P07-P08)
    "FMSynthesis",
    "RadarPolyphaseCode",
    "SpreadSpectrumRadar",
    # Molecular (P02, P05-P06)
    "LennardJonesPotential",
    "TersoffPotentialSiB",
    "TersoffPotentialSiC",
    # Engineering (P09-P11)
    "CircularAntennaArray",
    "DynamicEconomicDispatch",
    "HydrothermalScheduling",
    # Spacecraft trajectory (P12-P22)
    "Cassini2",
    "Messenger",
    "Cassini1",
    "GTOC1",
    "Rosetta",
    "Sagas",
    "Cassini2Tight",
    "MessengerTight",
    "GTOC1Tight",
    "RosettaTight",
    "SagasTight",
    # Collections
    "CEC2011_ALL",
    "CEC2011_SIMPLE",
    "CEC2011_SIGNAL",
    "CEC2011_MOLECULAR",
    "CEC2011_ENGINEERING",
    "CEC2011_SPACECRAFT",
]
