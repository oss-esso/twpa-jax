# Hybrid map productionization snapshot

This is the known-good development reference before repository cleanup.

| Mode | Valid gain points | Requested | Raw coverage | Runtime |
|---|---:|---:|---:|---:|
| Fast HB baseline | 187 | 400 | 46.75% | about 402 s |
| Slow HB + TD | 223 | 400 | 55.75% | about 3024 s |

The slow reference used 20 independent frequency columns. Every column reached a
TD bridge: 14 ended in persistent non-`PERIOD_1` states and 6 ended in transient
numerical blockers. No TD-periodic-to-gain handoff occurred naturally; the extra
36 valid points came from the bounded HB recovery ladder. Compact output was about
22 MB.

The separately validated 7.9 GHz physical-boundary reference is 11.40 uA as the
last working `PERIOD_1` point and 11.60 uA as the first persistent outside point;
the HB accessibility/fold event is about 11.30 uA.
