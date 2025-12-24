# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Time-series forecasting datasets."""

import numpy as np

# Lazy-loaded datasets
_airline_data = None
_energy_data = None
_sine_wave_data = None


def airline_data():
    """Classic airline passengers dataset (144 monthly samples).

    Monthly totals of international airline passengers from 1949-1960.
    A classic time-series benchmark with trend and seasonality.

    Returns
    -------
    tuple
        (X, y) where X is time indices and y is passenger counts.
    """
    global _airline_data
    if _airline_data is None:
        # Classic airline passengers data (1949-1960, monthly)
        passengers = np.array(
            [
                112,
                118,
                132,
                129,
                121,
                135,
                148,
                148,
                136,
                119,
                104,
                118,
                115,
                126,
                141,
                135,
                125,
                149,
                170,
                170,
                158,
                133,
                114,
                140,
                145,
                150,
                178,
                163,
                172,
                178,
                199,
                199,
                184,
                162,
                146,
                166,
                171,
                180,
                193,
                181,
                183,
                218,
                230,
                242,
                209,
                191,
                172,
                194,
                196,
                196,
                236,
                235,
                229,
                243,
                264,
                272,
                237,
                211,
                180,
                201,
                204,
                188,
                235,
                227,
                234,
                264,
                302,
                293,
                259,
                229,
                203,
                229,
                242,
                233,
                267,
                269,
                270,
                315,
                364,
                347,
                312,
                274,
                237,
                278,
                284,
                277,
                317,
                313,
                318,
                374,
                413,
                405,
                355,
                306,
                271,
                306,
                315,
                301,
                356,
                348,
                355,
                422,
                465,
                467,
                404,
                347,
                305,
                336,
                340,
                318,
                362,
                348,
                363,
                435,
                491,
                505,
                404,
                359,
                310,
                337,
                360,
                342,
                406,
                396,
                420,
                472,
                548,
                559,
                463,
                407,
                362,
                405,
                417,
                391,
                419,
                461,
                472,
                535,
                622,
                606,
                508,
                461,
                390,
                432,
            ],
            dtype=np.float64,
        )
        t = np.arange(len(passengers))
        _airline_data = (t.reshape(-1, 1), passengers)
    return _airline_data


def energy_data():
    """Synthetic energy consumption dataset (500 samples).

    Simulated hourly energy consumption with daily and weekly patterns.

    Returns
    -------
    tuple
        (X, y) where X is time indices and y is energy consumption.
    """
    global _energy_data
    if _energy_data is None:
        np.random.seed(42)
        n_samples = 500
        t = np.arange(n_samples)

        # Base consumption with trend
        base = 100 + 0.05 * t

        # Daily pattern (24-hour cycle)
        daily = 20 * np.sin(2 * np.pi * t / 24)

        # Weekly pattern (168-hour cycle)
        weekly = 10 * np.sin(2 * np.pi * t / 168)

        # Random noise
        noise = np.random.normal(0, 5, n_samples)

        consumption = base + daily + weekly + noise
        _energy_data = (t.reshape(-1, 1), consumption)
    return _energy_data


def sine_wave_data():
    """Synthetic sine wave dataset for baseline testing (200 samples).

    Simple periodic data with noise for quick testing.

    Returns
    -------
    tuple
        (X, y) where X is time indices and y is sine values with noise.
    """
    global _sine_wave_data
    if _sine_wave_data is None:
        np.random.seed(42)
        n_samples = 200
        t = np.arange(n_samples)

        # Simple sine wave with trend and noise
        y = 10 * np.sin(2 * np.pi * t / 20) + 0.1 * t + np.random.normal(0, 1, n_samples)

        _sine_wave_data = (t.reshape(-1, 1), y)
    return _sine_wave_data


# Registry for easy access
DATASETS = {
    "airline": airline_data,
    "energy": energy_data,
    "sine_wave": sine_wave_data,
}
