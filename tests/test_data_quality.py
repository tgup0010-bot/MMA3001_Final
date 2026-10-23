"""Unit tests for occupancy.data_quality, on manufactured series with known answers."""

import numpy as np
import pandas as pd

from occupancy.data_quality import hampel_flags, spike_decay_rates


def _series(values, start="2025-10-01", freq="10min"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.date_range(start, periods=len(values), freq=freq))


def test_hampel_flags_an_isolated_spike_and_nothing_else():
    rng = np.random.default_rng(0)
    values = 440 + rng.normal(0, 3, 200).round()
    values[100] = 700
    flags = hampel_flags(_series(values))
    assert flags.iloc[100]
    assert flags.sum() == 1


def test_hampel_ignores_small_wobbles_on_a_flat_signal():
    values = [440] * 50 + [444] + [440] * 50  # MAD is zero; 4 ppm is not an outlier
    assert not hampel_flags(_series(values)).any()


def test_decay_rate_recovers_a_known_ventilation_rate():
    true_rate = 4.0  # air changes per hour
    hours = np.arange(10) / 6.0  # 10-minute readings
    release = 300 * np.exp(-true_rate * hours)
    values = np.concatenate([np.full(10, 430.0), 430 + release, np.full(10, 430.0)])
    rates = spike_decay_rates(_series(values))
    assert len(rates) == 1
    assert abs(rates[0] - true_rate) < 1e-6


def test_decay_events_with_a_missing_reading_are_skipped():
    hours = np.arange(10) / 6.0
    values = np.concatenate([np.full(10, 430.0), 430 + 300 * np.exp(-4 * hours)])
    s = _series(values).drop(index=_series(values).index[12])  # gap right after the peak
    assert len(spike_decay_rates(s)) == 0
