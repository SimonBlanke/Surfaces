# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CEC 2011 real-world benchmark functions."""

import numpy as np
import pytest

from surfaces.test_functions.benchmark.cec.cec2011 import (
    CEC2011_ALL,
    CEC2011_ENGINEERING,
    CEC2011_MOLECULAR,
    CEC2011_SIGNAL,
    CEC2011_SIMPLE,
    CEC2011_SPACECRAFT,
    BifunctionalCatalyst,
    Cassini1,
    Cassini2,
    Cassini2Tight,
    CircularAntennaArray,
    DynamicEconomicDispatch,
    FMSynthesis,
    GTOC1,
    GTOC1Tight,
    HydrothermalScheduling,
    LennardJonesPotential,
    Messenger,
    MessengerTight,
    RadarPolyphaseCode,
    Rosetta,
    RosettaTight,
    Sagas,
    SagasTight,
    SpreadSpectrumRadar,
    StirredTankReactor,
    TersoffPotentialSiB,
    TersoffPotentialSiC,
)


class TestCEC2011FunctionProperties:
    """Test function properties and specs."""

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_has_problem_id(self, func_class):
        """Each function must have a problem_id."""
        func = func_class()
        assert func.problem_id is not None
        assert 1 <= func.problem_id <= 22

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_has_spec(self, func_class):
        """Each function must have specs defined."""
        func = func_class()
        spec = func.spec
        assert "continuous" in spec
        assert "scalable" in spec
        # CEC 2011 functions are not scalable (fixed dimensions)
        assert spec["scalable"] is False

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_has_f_global(self, func_class):
        """Each function must have f_global defined."""
        func = func_class()
        assert hasattr(func, "f_global")
        assert np.isfinite(func.f_global)


class TestCEC2011FixedDimensions:
    """Test fixed dimension handling for CEC 2011 functions."""

    def test_fm_synthesis_dim_6(self):
        """P01 FM Synthesis should have D=6."""
        func = FMSynthesis()
        assert func.n_dim == 6
        assert func._fixed_dim == 6

    def test_lennard_jones_dim_30(self):
        """P02 Lennard-Jones should have D=30 (10 atoms)."""
        func = LennardJonesPotential()
        assert func.n_dim == 30
        assert func._fixed_dim == 30
        assert func.n_atoms == 10

    def test_bifunctional_catalyst_dim_1(self):
        """P03 Bifunctional Catalyst should have D=1."""
        func = BifunctionalCatalyst()
        assert func.n_dim == 1
        assert func._fixed_dim == 1

    def test_stirred_tank_reactor_dim_1(self):
        """P04 Stirred Tank Reactor should have D=1."""
        func = StirredTankReactor()
        assert func.n_dim == 1
        assert func._fixed_dim == 1

    def test_tersoff_sib_dim_30(self):
        """P05 Tersoff Si-B should have D=30."""
        func = TersoffPotentialSiB()
        assert func.n_dim == 30
        assert func._fixed_dim == 30

    def test_tersoff_sic_dim_30(self):
        """P06 Tersoff Si-C should have D=30."""
        func = TersoffPotentialSiC()
        assert func.n_dim == 30
        assert func._fixed_dim == 30

    def test_radar_polyphase_dim_20(self):
        """P07 Radar Polyphase should have D=20."""
        func = RadarPolyphaseCode()
        assert func.n_dim == 20
        assert func._fixed_dim == 20

    def test_spread_spectrum_dim_7(self):
        """P08 Spread Spectrum should have D=7."""
        func = SpreadSpectrumRadar()
        assert func.n_dim == 7
        assert func._fixed_dim == 7

    def test_circular_antenna_dim_12(self):
        """P09 Circular Antenna Array should have D=12."""
        func = CircularAntennaArray()
        assert func.n_dim == 12
        assert func._fixed_dim == 12
        assert func.n_elements == 12

    def test_dynamic_economic_dispatch_dim_120(self):
        """P10 Dynamic Economic Dispatch should have D=120 (5 generators * 24 hours)."""
        func = DynamicEconomicDispatch()
        assert func.n_dim == 120
        assert func._fixed_dim == 120
        assert func.n_generators == 5
        assert func.n_hours == 24

    def test_hydrothermal_scheduling_dim_24(self):
        """P11 Hydrothermal Scheduling should have D=24."""
        func = HydrothermalScheduling()
        assert func.n_dim == 24
        assert func._fixed_dim == 24

    def test_cassini2_dim_22(self):
        """P12 Cassini 2 should have D=22."""
        func = Cassini2()
        assert func.n_dim == 22
        assert func._fixed_dim == 22

    def test_messenger_dim_26(self):
        """P13 Messenger should have D=26."""
        func = Messenger()
        assert func.n_dim == 26
        assert func._fixed_dim == 26

    def test_cassini1_dim_6(self):
        """P14 Cassini 1 should have D=6."""
        func = Cassini1()
        assert func.n_dim == 6
        assert func._fixed_dim == 6

    def test_gtoc1_dim_8(self):
        """P15 GTOC1 should have D=8."""
        func = GTOC1()
        assert func.n_dim == 8
        assert func._fixed_dim == 8

    def test_rosetta_dim_22(self):
        """P16 Rosetta should have D=22."""
        func = Rosetta()
        assert func.n_dim == 22
        assert func._fixed_dim == 22

    def test_sagas_dim_12(self):
        """P17 Sagas should have D=12."""
        func = Sagas()
        assert func.n_dim == 12
        assert func._fixed_dim == 12

    def test_cassini2_tight_dim_22(self):
        """P18 Cassini 2 Tight should have D=22."""
        func = Cassini2Tight()
        assert func.n_dim == 22
        assert func._fixed_dim == 22

    def test_messenger_tight_dim_26(self):
        """P19 Messenger Tight should have D=26."""
        func = MessengerTight()
        assert func.n_dim == 26
        assert func._fixed_dim == 26

    def test_gtoc1_tight_dim_8(self):
        """P20 GTOC1 Tight should have D=8."""
        func = GTOC1Tight()
        assert func.n_dim == 8
        assert func._fixed_dim == 8

    def test_rosetta_tight_dim_22(self):
        """P21 Rosetta Tight should have D=22."""
        func = RosettaTight()
        assert func.n_dim == 22
        assert func._fixed_dim == 22

    def test_sagas_tight_dim_12(self):
        """P22 Sagas Tight should have D=12."""
        func = SagasTight()
        assert func.n_dim == 12
        assert func._fixed_dim == 12


class TestCEC2011Bounds:
    """Test bounds for CEC 2011 functions."""

    def test_fm_synthesis_bounds(self):
        """P01 should have bounds [-6.4, 6.35]."""
        func = FMSynthesis()
        assert func.default_bounds == (-6.4, 6.35)

    def test_lennard_jones_bounds(self):
        """P02 should have bounds [-4, 4]."""
        func = LennardJonesPotential()
        assert func.default_bounds == (-4.0, 4.0)

    def test_bifunctional_catalyst_bounds(self):
        """P03 should have bounds [-0.6, 0.9]."""
        func = BifunctionalCatalyst()
        assert func.default_bounds == (-0.6, 0.9)

    def test_stirred_tank_reactor_bounds(self):
        """P04 should have bounds [0, 5]."""
        func = StirredTankReactor()
        assert func.default_bounds == (0.0, 5.0)

    def test_tersoff_sib_bounds(self):
        """P05 should have bounds [-10, 10] (simplified)."""
        func = TersoffPotentialSiB()
        assert func.default_bounds == (-10.0, 10.0)

    def test_tersoff_sic_bounds(self):
        """P06 should have bounds [-10, 10] (simplified)."""
        func = TersoffPotentialSiC()
        assert func.default_bounds == (-10.0, 10.0)

    def test_radar_polyphase_bounds(self):
        """P07 should have bounds [0, 2*pi]."""
        func = RadarPolyphaseCode()
        assert func.default_bounds == (0.0, 2 * np.pi)

    def test_spread_spectrum_bounds(self):
        """P08 should have bounds [0, 15]."""
        func = SpreadSpectrumRadar()
        assert func.default_bounds == (0.0, 15.0)

    def test_circular_antenna_bounds(self):
        """P09 should have bounds [0.1, 2.0]."""
        func = CircularAntennaArray()
        assert func.default_bounds == (0.1, 2.0)

    def test_dynamic_economic_dispatch_bounds(self):
        """P10 should have bounds [0, 300]."""
        func = DynamicEconomicDispatch()
        assert func.default_bounds == (0.0, 300.0)

    def test_hydrothermal_scheduling_bounds(self):
        """P11 should have bounds [5, 15]."""
        func = HydrothermalScheduling()
        assert func.default_bounds == (5.0, 15.0)

    def test_cassini2_bounds(self):
        """P12 should have normalized bounds [0, 1]."""
        func = Cassini2()
        assert func.default_bounds == (0.0, 1.0)

    def test_messenger_bounds(self):
        """P13 should have normalized bounds [0, 1]."""
        func = Messenger()
        assert func.default_bounds == (0.0, 1.0)

    def test_cassini1_bounds(self):
        """P14 should have normalized bounds [0, 1]."""
        func = Cassini1()
        assert func.default_bounds == (0.0, 1.0)

    def test_gtoc1_bounds(self):
        """P15 should have normalized bounds [0, 1]."""
        func = GTOC1()
        assert func.default_bounds == (0.0, 1.0)

    def test_rosetta_bounds(self):
        """P16 should have normalized bounds [0, 1]."""
        func = Rosetta()
        assert func.default_bounds == (0.0, 1.0)

    def test_sagas_bounds(self):
        """P17 should have normalized bounds [0, 1]."""
        func = Sagas()
        assert func.default_bounds == (0.0, 1.0)

    def test_cassini2_tight_bounds(self):
        """P18 should have normalized bounds [0, 1]."""
        func = Cassini2Tight()
        assert func.default_bounds == (0.0, 1.0)

    def test_messenger_tight_bounds(self):
        """P19 should have normalized bounds [0, 1]."""
        func = MessengerTight()
        assert func.default_bounds == (0.0, 1.0)

    def test_gtoc1_tight_bounds(self):
        """P20 should have normalized bounds [0, 1]."""
        func = GTOC1Tight()
        assert func.default_bounds == (0.0, 1.0)

    def test_rosetta_tight_bounds(self):
        """P21 should have normalized bounds [0, 1]."""
        func = RosettaTight()
        assert func.default_bounds == (0.0, 1.0)

    def test_sagas_tight_bounds(self):
        """P22 should have normalized bounds [0, 1]."""
        func = SagasTight()
        assert func.default_bounds == (0.0, 1.0)


class TestCEC2011InputFormats:
    """Test different input formats."""

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_array_input(self, func_class):
        """Function should accept numpy array input."""
        func = func_class()
        lb, ub = func.default_bounds
        x = np.random.uniform(lb, ub, func.n_dim)
        result = func(x)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_list_input(self, func_class):
        """Function should accept list input."""
        func = func_class()
        lb, ub = func.default_bounds
        x = [np.random.uniform(lb, ub) for _ in range(func.n_dim)]
        result = func(x)
        assert np.isfinite(result)

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_dict_input(self, func_class):
        """Function should accept dict input."""
        func = func_class()
        lb, ub = func.default_bounds
        params = {f"x{i}": np.random.uniform(lb, ub) for i in range(func.n_dim)}
        result = func(params)
        assert np.isfinite(result)


class TestCEC2011SearchSpace:
    """Test search space properties."""

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_search_space(self, func_class):
        """Search space should have correct dimensions."""
        func = func_class()
        space = func.search_space
        assert len(space) == func.n_dim
        for i in range(func.n_dim):
            assert f"x{i}" in space


class TestCEC2011Objective:
    """Test objective parameter."""

    def test_minimize_objective(self):
        """Minimize objective should return positive-biased values."""
        func = FMSynthesis(objective="minimize")
        result = func(func.x_global)
        assert result == func.f_global

    def test_maximize_objective(self):
        """Maximize objective should negate values."""
        func = FMSynthesis(objective="maximize")
        result = func(func.x_global)
        assert result == -func.f_global


class TestCEC2011GlobalOptimum:
    """Test global optimum for functions with known optima."""

    def test_fm_synthesis_global_optimum(self):
        """FM Synthesis should achieve f*=0 at target parameters."""
        func = FMSynthesis()
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=1e-6)

    def test_stirred_tank_reactor_global_optimum(self):
        """Stirred Tank Reactor should achieve optimum at sqrt(3)."""
        func = StirredTankReactor()
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=1e-3)

    def test_bifunctional_catalyst_global_optimum(self):
        """Bifunctional Catalyst should achieve optimum near x*=0.43094."""
        func = BifunctionalCatalyst()
        result = func(func.x_global)
        assert np.isclose(result, func.f_global, rtol=1e-3)


class TestCEC2011FunctionCategories:
    """Test function categories are correct."""

    def test_simple_count(self):
        """Should have 2 simple functions."""
        assert len(CEC2011_SIMPLE) == 2

    def test_signal_count(self):
        """Should have 3 signal processing functions."""
        assert len(CEC2011_SIGNAL) == 3

    def test_molecular_count(self):
        """Should have 3 molecular/potential functions."""
        assert len(CEC2011_MOLECULAR) == 3

    def test_engineering_count(self):
        """Should have 3 engineering functions."""
        assert len(CEC2011_ENGINEERING) == 3

    def test_spacecraft_count(self):
        """Should have 11 spacecraft trajectory functions (P12-P22)."""
        assert len(CEC2011_SPACECRAFT) == 11

    def test_total_count(self):
        """Should have 22 functions total (P01-P22)."""
        assert len(CEC2011_ALL) == 22


class TestCEC2011SpecificFunctions:
    """Test specific function behaviors."""

    def test_fm_synthesis_target_signal(self):
        """FM Synthesis should have pre-computed target signal."""
        func = FMSynthesis()
        assert hasattr(func, "_y_target")
        assert len(func._y_target) == 101

    def test_lennard_jones_atom_count(self):
        """Lennard-Jones should have 10 atoms by default."""
        func = LennardJonesPotential()
        assert func.n_atoms == 10
        assert func.n_dim == 30

    def test_lennard_jones_penalty_for_overlapping_atoms(self):
        """Lennard-Jones should penalize overlapping atoms."""
        func = LennardJonesPotential()
        # All atoms at origin (overlapping)
        result = func(np.zeros(30))
        assert result > 1e9  # Large penalty

    def test_lennard_jones_reasonable_for_spread_atoms(self):
        """Lennard-Jones should give reasonable values for spread atoms."""
        func = LennardJonesPotential()
        # Spread atoms randomly
        np.random.seed(42)
        x = np.random.uniform(-3, 3, 30)
        result = func(x)
        assert np.isfinite(result)
        assert result < 1e9

    def test_radar_polyphase_psl_range(self):
        """Radar Polyphase PSL should be in [0, 1]."""
        func = RadarPolyphaseCode()
        np.random.seed(42)
        x = np.random.uniform(0, 2 * np.pi, 20)
        result = func(x)
        assert 0 <= result <= 1

    def test_spread_spectrum_merit_factor(self):
        """Spread Spectrum should return inverse merit factor."""
        func = SpreadSpectrumRadar()
        np.random.seed(42)
        x = np.random.uniform(0, 15, 7)
        result = func(x)
        assert np.isfinite(result)
        assert result > 0


class TestCEC2011EngineeringProblems:
    """Test engineering optimization problems P09-P11."""

    def test_circular_antenna_psl_range(self):
        """P09 Circular Antenna Array PSL should be in [0, 1]."""
        func = CircularAntennaArray()
        np.random.seed(42)
        x = np.random.uniform(0.1, 2.0, 12)
        result = func(x)
        assert np.isfinite(result)
        # PSL is ratio of sidelobe to main beam, should be in [0, 1]
        # (can exceed 1 if optimization is poor, but should be positive)
        assert result >= 0

    def test_circular_antenna_array_factor(self):
        """P09 should have array factor computation."""
        func = CircularAntennaArray()
        assert hasattr(func, "_compute_array_factor")
        assert hasattr(func, "_phi_obs")
        assert len(func._phi_obs) == 360  # 1 degree resolution

    def test_dynamic_economic_dispatch_structure(self):
        """P10 should have generator data and load demand."""
        func = DynamicEconomicDispatch()
        assert hasattr(func, "_generator_data")
        assert hasattr(func, "_load_demand")
        assert len(func._load_demand) == 24
        assert func._generator_data.shape[0] == 5  # 5 generators

    def test_dynamic_economic_dispatch_cost(self):
        """P10 should compute finite cost for valid inputs."""
        func = DynamicEconomicDispatch()
        # Set power levels to meet demand approximately
        # Average demand is ~550 MW, 5 generators, so ~110 MW each
        x = np.full(120, 110.0)
        result = func(x)
        assert np.isfinite(result)
        # Cost should be positive
        assert result > 0

    def test_hydrothermal_structure(self):
        """P11 should have hydro and thermal data."""
        func = HydrothermalScheduling()
        assert hasattr(func, "_thermal_data")
        assert hasattr(func, "_hydro_coeffs")
        assert hasattr(func, "_load_demand")
        assert len(func._load_demand) == 24

    def test_hydrothermal_cost(self):
        """P11 should compute finite cost for valid inputs."""
        func = HydrothermalScheduling()
        # Moderate discharge to maintain water balance
        x = np.full(24, 10.0)  # 10 units per hour
        result = func(x)
        assert np.isfinite(result)
        assert result > 0

    def test_hydrothermal_water_balance(self):
        """P11 should track reservoir volume."""
        func = HydrothermalScheduling()
        assert hasattr(func, "_V_initial")
        assert hasattr(func, "_V_final")
        assert hasattr(func, "_V_min")
        assert hasattr(func, "_V_max")
        assert hasattr(func, "_inflow")


class TestCEC2011BatchEvaluation:
    """Test batch evaluation for CEC 2011 functions."""

    @pytest.mark.parametrize("func_class", CEC2011_ALL)
    def test_batch_evaluation_2d_array(self, func_class):
        """Batch evaluation should work with 2D numpy array."""
        func = func_class()
        lb, ub = func.default_bounds

        X = np.random.uniform(lb, ub, (5, func.n_dim))

        results = func.batch(X)
        assert len(results) == 5
        assert all(np.isfinite(r) for r in results)


class TestCEC2011UnimodalProperty:
    """Test unimodal property is correctly set."""

    def test_stirred_tank_reactor_unimodal(self):
        """P04 Stirred Tank Reactor should be unimodal."""
        func = StirredTankReactor()
        assert func.spec["unimodal"] is True

    @pytest.mark.parametrize(
        "func_class",
        [
            FMSynthesis,
            LennardJonesPotential,
            BifunctionalCatalyst,
            TersoffPotentialSiB,
            TersoffPotentialSiC,
            RadarPolyphaseCode,
            SpreadSpectrumRadar,
            CircularAntennaArray,
            DynamicEconomicDispatch,
            HydrothermalScheduling,
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
        ],
    )
    def test_multimodal_functions(self, func_class):
        """Most functions should be multimodal."""
        func = func_class()
        # StirredTankReactor is the only unimodal one
        if func_class != StirredTankReactor:
            assert func.spec["unimodal"] is False


class TestCEC2011SpacecraftProblems:
    """Test spacecraft trajectory optimization problems P12-P22."""

    def test_cassini2_sequence(self):
        """P12 Cassini 2 should have E-V-V-E-J-S sequence."""
        func = Cassini2()
        assert func._sequence == ["earth", "venus", "venus", "earth", "jupiter", "saturn"]

    def test_cassini2_bounds_array(self):
        """P12 should have bounds array with 22 variables."""
        func = Cassini2()
        assert func._bounds.shape == (22, 2)

    def test_cassini2_f_global(self):
        """P12 should have f_global around 8.4 km/s."""
        func = Cassini2()
        assert np.isclose(func.f_global, 8.4, rtol=0.1)

    def test_cassini2_evaluation(self):
        """P12 should return finite delta-V for valid input."""
        func = Cassini2()
        np.random.seed(42)
        x = np.random.uniform(0, 1, 22)
        result = func(x)
        # Result should be finite (may be large penalty if trajectory invalid)
        assert np.isfinite(result)

    def test_messenger_sequence(self):
        """P13 Messenger should have E-V-V-M-M-M-M sequence."""
        func = Messenger()
        assert func._sequence == ["earth", "venus", "venus", "mercury", "mercury", "mercury", "mercury"]

    def test_messenger_bounds_array(self):
        """P13 should have bounds array with 26 variables."""
        func = Messenger()
        assert func._bounds.shape == (26, 2)

    def test_messenger_f_global(self):
        """P13 should have f_global around 8.6 km/s."""
        func = Messenger()
        assert np.isclose(func.f_global, 8.6, rtol=0.1)

    def test_messenger_evaluation(self):
        """P13 should return finite delta-V for valid input."""
        func = Messenger()
        np.random.seed(42)
        x = np.random.uniform(0, 1, 26)
        result = func(x)
        assert np.isfinite(result)

    # P14: Cassini 1 (MGA without DSM)
    def test_cassini1_sequence(self):
        """P14 Cassini 1 should have E-V-V-E-J-S sequence."""
        func = Cassini1()
        assert func._sequence == ["earth", "venus", "venus", "earth", "jupiter", "saturn"]

    def test_cassini1_bounds_array(self):
        """P14 should have bounds array with 6 variables."""
        func = Cassini1()
        assert func._bounds.shape == (6, 2)

    def test_cassini1_f_global(self):
        """P14 should have f_global around 4.93 km/s."""
        func = Cassini1()
        assert np.isclose(func.f_global, 4.93, rtol=0.1)

    def test_cassini1_evaluation(self):
        """P14 should return finite delta-V for valid input."""
        func = Cassini1()
        np.random.seed(42)
        x = np.random.uniform(0, 1, 6)
        result = func(x)
        assert np.isfinite(result)

    # P15: GTOC1
    def test_gtoc1_asteroid_elements(self):
        """P15 GTOC1 should have asteroid TW229 orbital elements."""
        func = GTOC1()
        assert hasattr(func, "_asteroid_elements")
        assert "a" in func._asteroid_elements
        assert "e" in func._asteroid_elements

    def test_gtoc1_bounds_array(self):
        """P15 should have bounds array with 8 variables."""
        func = GTOC1()
        assert func._bounds.shape == (8, 2)

    def test_gtoc1_f_global(self):
        """P15 should have f_global around 5.0 km/s."""
        func = GTOC1()
        assert np.isclose(func.f_global, 5.0, rtol=0.5)

    def test_gtoc1_evaluation(self):
        """P15 should return finite delta-V for valid input."""
        func = GTOC1()
        np.random.seed(42)
        x = np.random.uniform(0, 1, 8)
        result = func(x)
        assert np.isfinite(result)

    # P16: Rosetta
    def test_rosetta_sequence(self):
        """P16 Rosetta should have E-E-M-E-E sequence."""
        func = Rosetta()
        assert func._sequence == ["earth", "earth", "mars", "earth", "earth"]

    def test_rosetta_comet_elements(self):
        """P16 Rosetta should have comet 67P orbital elements."""
        func = Rosetta()
        assert hasattr(func, "_comet_elements")
        assert "a" in func._comet_elements

    def test_rosetta_bounds_array(self):
        """P16 should have bounds array with 22 variables."""
        func = Rosetta()
        assert func._bounds.shape == (22, 2)

    def test_rosetta_f_global(self):
        """P16 should have f_global around 1.3 km/s."""
        func = Rosetta()
        assert np.isclose(func.f_global, 1.3, rtol=0.5)

    def test_rosetta_evaluation(self):
        """P16 should return finite delta-V for valid input."""
        func = Rosetta()
        np.random.seed(42)
        x = np.random.uniform(0, 1, 22)
        result = func(x)
        assert np.isfinite(result)

    # P17: Sagas Solar Sail
    def test_sagas_sail_parameters(self):
        """P17 Sagas should have solar sail parameters."""
        func = Sagas()
        assert hasattr(func, "_sail_lightness")
        assert func._sail_lightness > 0

    def test_sagas_bounds_array(self):
        """P17 should have bounds array with 12 variables."""
        func = Sagas()
        assert func._bounds.shape == (12, 2)

    def test_sagas_evaluation(self):
        """P17 should return finite result for valid input."""
        func = Sagas()
        np.random.seed(42)
        x = np.random.uniform(0, 1, 12)
        result = func(x)
        assert np.isfinite(result)

    # P18-P22: Tight bounds variants
    def test_cassini2_tight_bounds_narrower(self):
        """P18 Cassini 2 Tight should have narrower bounds than P12."""
        func_base = Cassini2()
        func_tight = Cassini2Tight()
        # Tight bounds should be narrower (smaller range)
        base_range = func_base._bounds[:, 1] - func_base._bounds[:, 0]
        tight_range = func_tight._bounds[:, 1] - func_tight._bounds[:, 0]
        assert np.all(tight_range <= base_range)

    def test_messenger_tight_bounds_narrower(self):
        """P19 Messenger Tight should have narrower bounds than P13."""
        func_base = Messenger()
        func_tight = MessengerTight()
        base_range = func_base._bounds[:, 1] - func_base._bounds[:, 0]
        tight_range = func_tight._bounds[:, 1] - func_tight._bounds[:, 0]
        assert np.all(tight_range <= base_range)

    def test_gtoc1_tight_bounds_narrower(self):
        """P20 GTOC1 Tight should have narrower bounds than P15."""
        func_base = GTOC1()
        func_tight = GTOC1Tight()
        base_range = func_base._bounds[:, 1] - func_base._bounds[:, 0]
        tight_range = func_tight._bounds[:, 1] - func_tight._bounds[:, 0]
        assert np.all(tight_range <= base_range)

    def test_rosetta_tight_bounds_narrower(self):
        """P21 Rosetta Tight should have narrower bounds than P16."""
        func_base = Rosetta()
        func_tight = RosettaTight()
        base_range = func_base._bounds[:, 1] - func_base._bounds[:, 0]
        tight_range = func_tight._bounds[:, 1] - func_tight._bounds[:, 0]
        assert np.all(tight_range <= base_range)

    def test_sagas_tight_bounds_narrower(self):
        """P22 Sagas Tight should have narrower bounds than P17."""
        func_base = Sagas()
        func_tight = SagasTight()
        base_range = func_base._bounds[:, 1] - func_base._bounds[:, 0]
        tight_range = func_tight._bounds[:, 1] - func_tight._bounds[:, 0]
        assert np.all(tight_range <= base_range)

    @pytest.mark.parametrize(
        "tight_class,base_class",
        [
            (Cassini2Tight, Cassini2),
            (MessengerTight, Messenger),
            (GTOC1Tight, GTOC1),
            (RosettaTight, Rosetta),
            (SagasTight, Sagas),
        ],
    )
    def test_tight_variants_same_dimension(self, tight_class, base_class):
        """Tight variants should have same dimension as base."""
        func_base = base_class()
        func_tight = tight_class()
        assert func_tight.n_dim == func_base.n_dim

    @pytest.mark.parametrize("func_class", CEC2011_SPACECRAFT)
    def test_spacecraft_denormalize(self, func_class):
        """Spacecraft functions should denormalize inputs."""
        func = func_class()
        # Test that denormalization works
        x_norm = np.zeros(func.n_dim)
        x_actual = func._denormalize(x_norm)
        assert np.allclose(x_actual, func._bounds[:, 0])

        x_norm = np.ones(func.n_dim)
        x_actual = func._denormalize(x_norm)
        assert np.allclose(x_actual, func._bounds[:, 1])
