from argparse import Namespace

from scripts.run_gain_map import signal_ghz_for, spectrum_offsets_mhz


def test_default_spectrum_offsets_include_headline_signal() -> None:
    args = Namespace(
        signal_ghz=None,
        signal_detuning_mhz=100.0,
        signal_offset_start_mhz=100.0,
        signal_offset_step_mhz=500.0,
        signal_offset_count_per_side=5,
    )

    pump_freq_ghz = 7.25
    target = signal_ghz_for(pump_freq_ghz, args)
    spectrum_freqs = [
        pump_freq_ghz + offset_mhz / 1000.0
        for offset_mhz in spectrum_offsets_mhz(args)
    ]

    assert target in spectrum_freqs
