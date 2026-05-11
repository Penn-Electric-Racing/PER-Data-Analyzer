import numpy as np
import pytest
from scipy.signal import butter

from perda.core_data_structures.data_instance import DataInstance
from perda.units import Timescale
from perda.utils.filtering import *


def test_apply_sos_filter_basic():
    sos = butter(4, 0.1, btype="low", output="sos")
    signal = np.ones(200, dtype=np.float64)
    result = apply_sos_filter(signal, sos, order=4)
    assert result is not None
    np.testing.assert_allclose(result, 1.0, atol=1e-6)


def test_apply_sos_filter_preserves_nan_positions():
    sos = butter(4, 0.1, btype="low", output="sos")
    signal = np.ones(200, dtype=np.float64)
    signal[10] = np.nan
    signal[50] = np.nan
    result = apply_sos_filter(signal, sos, order=4)
    assert result is not None
    assert np.isnan(result[10])
    assert np.isnan(result[50])
    assert not np.any(np.isnan(np.delete(result, [10, 50])))


def test_apply_sos_filter_returns_none_when_too_few_samples():
    sos = butter(4, 0.1, btype="low", output="sos")
    # order=4 → min_samples = 3*(2*4+1) = 27; use only 2 valid points
    signal = np.full(5, np.nan, dtype=np.float64)
    signal[0] = 1.0
    signal[1] = 1.0
    result = apply_sos_filter(signal, sos, order=4)
    assert result is None


@pytest.mark.parametrize("cutoff_hz", [5.0, 10.0, 20.0])
def test_lowpass_attenuates_high_frequency(cutoff_hz: float, sine_di_factory):
    """A signal at 4× cutoff should be substantially attenuated in steady-state."""
    di = sine_di_factory(freq_hz=cutoff_hz * 4, n=2000)
    result = lowpass_filter(di, cutoff_hz=cutoff_hz)
    assert isinstance(result, DataInstance)
    # Compare RMS over middle 50% to avoid filter transients at signal edges
    mid = slice(len(di.value_np) // 4, 3 * len(di.value_np) // 4)
    rms_in = np.sqrt(np.mean(di.value_np[mid] ** 2))
    rms_out = np.sqrt(np.mean(result.value_np[mid] ** 2))
    assert rms_out < rms_in * 0.1


@pytest.mark.parametrize("cutoff_hz", [5.0, 10.0, 20.0])
def test_lowpass_passes_low_frequency(cutoff_hz: float, sine_di_factory):
    """A signal well below the cutoff should pass through nearly unchanged."""
    di = sine_di_factory(freq_hz=cutoff_hz / 4)
    result = lowpass_filter(di, cutoff_hz=cutoff_hz)
    assert isinstance(result, DataInstance)
    np.testing.assert_allclose(result.value_np, di.value_np, atol=0.05)


def test_lowpass_preserves_metadata(sine_di_factory):
    di_base = sine_di_factory(freq_hz=5.0)
    di = DataInstance(
        timestamp_np=di_base.timestamp_np,
        value_np=di_base.value_np,
        label="myvar",
        var_id=42,
        cpp_name="pcm.myvar",
    )
    result = lowpass_filter(di, cutoff_hz=20.0)
    assert isinstance(result, DataInstance)
    assert result.label == "myvar"
    assert result.var_id == 42
    assert result.cpp_name == "pcm.myvar"
    np.testing.assert_array_equal(result.timestamp_np, di.timestamp_np)


def test_lowpass_list_input_returns_list(sine_di_factory):
    di_a = sine_di_factory(freq_hz=5.0, label="a", var_id=1)
    di_b = sine_di_factory(freq_hz=5.0, label="b", var_id=2)
    results = lowpass_filter([di_a, di_b], cutoff_hz=20.0)
    assert isinstance(results, list)
    assert len(results) == 2


def test_lowpass_skips_when_cutoff_above_nyquist(sine_di_factory, capsys):
    """When cutoff >= Nyquist the original DataInstance is returned unchanged."""
    di = sine_di_factory(freq_hz=5.0)
    # Default fs=200 Hz → Nyquist=100 Hz; use 150 Hz cutoff
    result = lowpass_filter(di, cutoff_hz=150.0)
    assert isinstance(result, DataInstance)
    np.testing.assert_array_equal(result.value_np, di.value_np)
    captured = capsys.readouterr()
    assert "Nyquist" in captured.out


def test_lowpass_timescale_ms_vs_us(sine_di_ms, sine_di_us):
    """Same physical signal filtered identically regardless of timestamp unit."""
    result_ms = lowpass_filter(
        sine_di_ms, cutoff_hz=20.0, source_time_unit=Timescale.MS
    )
    result_us = lowpass_filter(
        sine_di_us, cutoff_hz=20.0, source_time_unit=Timescale.US
    )
    assert isinstance(result_ms, DataInstance)
    assert isinstance(result_us, DataInstance)
    np.testing.assert_allclose(result_ms.value_np, result_us.value_np, atol=1e-10)


def test_zscore_removes_spike(uniform_di):
    """A single large spike should be removed and interpolated."""
    values = uniform_di.value_np.copy()
    values[250] = 1000.0
    di = DataInstance(
        timestamp_np=uniform_di.timestamp_np,
        value_np=values,
        label="spike",
        var_id=1,
    )
    result = zscore_filter(
        di, window_s=0.5, threshold=3.0, source_time_unit=Timescale.MS
    )
    assert isinstance(result, DataInstance)
    assert result.value_np[250] < 10.0


def test_zscore_keeps_clean_signal(uniform_di):
    """A clean constant signal should be unchanged."""
    result = zscore_filter(
        uniform_di, window_s=0.5, threshold=3.0, source_time_unit=Timescale.MS
    )
    assert isinstance(result, DataInstance)
    np.testing.assert_allclose(result.value_np, uniform_di.value_np, atol=1e-10)


@pytest.mark.parametrize("n_spikes", [1, 3, 5])
def test_zscore_removes_multiple_spikes(n_spikes: int, uniform_di):
    values = uniform_di.value_np.copy()
    spike_indices = np.linspace(50, 450, n_spikes, dtype=int)
    values[spike_indices] = 1000.0
    di = DataInstance(
        timestamp_np=uniform_di.timestamp_np,
        value_np=values,
        label="spikes",
        var_id=1,
    )
    result = zscore_filter(
        di, window_s=0.5, threshold=3.0, source_time_unit=Timescale.MS
    )
    assert isinstance(result, DataInstance)
    assert np.all(result.value_np[spike_indices] < 10.0)


def test_zscore_preserves_timestamps(uniform_di):
    result = zscore_filter(uniform_di)
    np.testing.assert_array_equal(result.timestamp_np, uniform_di.timestamp_np)


def test_zscore_list_input_returns_list(uniform_di):
    di_a = DataInstance(
        timestamp_np=uniform_di.timestamp_np,
        value_np=uniform_di.value_np.copy(),
        label="a",
        var_id=1,
    )
    di_b = DataInstance(
        timestamp_np=uniform_di.timestamp_np,
        value_np=uniform_di.value_np.copy(),
        label="b",
        var_id=2,
    )
    results = zscore_filter([di_a, di_b])
    assert isinstance(results, list)
    assert len(results) == 2


@pytest.mark.parametrize("freq_hz", [5.0, 10.0, 20.0, 50.0])
def test_compute_fft_peak_at_correct_frequency(freq_hz: float, sine_di_factory):
    """FFT peak should be at the injected sine frequency."""
    di = sine_di_factory(freq_hz=freq_hz, n=2000)
    freqs, mags = compute_fft(di, source_time_unit=Timescale.MS)
    peak_freq = freqs[np.argmax(mags)]
    assert abs(peak_freq - freq_hz) < 1.0


def test_compute_fft_output_shapes_match(sine_di_factory):
    di = sine_di_factory(freq_hz=10.0)
    freqs, mags = compute_fft(di, source_time_unit=Timescale.MS)
    assert freqs.shape == mags.shape
    assert len(freqs) > 0


def test_compute_fft_drops_nan(sine_di_factory):
    """NaN values should be dropped without raising an error."""
    di = sine_di_factory(freq_hz=10.0)
    values_with_nan = di.value_np.copy()
    values_with_nan[[5, 10, 15]] = np.nan
    di_nan = DataInstance(
        timestamp_np=di.timestamp_np,
        value_np=values_with_nan,
        label="nan_sine",
        var_id=1,
    )
    freqs, mags = compute_fft(di_nan, source_time_unit=Timescale.MS)
    assert not np.any(np.isnan(mags))
    assert len(freqs) > 0


def test_compute_fft_dc_suppressed(uniform_di):
    """Mean subtraction should make DC (0 Hz) component near zero."""
    freqs, mags = compute_fft(uniform_di, source_time_unit=Timescale.MS)
    dc_idx = np.argmin(np.abs(freqs))
    assert mags[dc_idx] < 1e-6
