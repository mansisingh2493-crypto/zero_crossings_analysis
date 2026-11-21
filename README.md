Chirp-mass estimation from zero-crossing analysis.

Basic analysis extended with:
  * fixed optimal SNR (default 50, O5-like) by rescaling the distance
  * many noise realisations per parameter set
  * automatic cycle-window selection (no hand-tuned start/end)
  * outlier rejection on the frequency track, weighted fit, fit uncertainties
  * summary table + presentation plots

Usage:   python chirp_mass_pipeline.py                 (reads zca_results.xlsx)
         python chirp_mass_pipeline.py other_table.xlsx
Input: the zca_results.xlsx layout (group label | m1 m2 sz1 sy1 sx1 sz2 sy2 sx2 ...;
blank masses inherit the row above within a group, blank spins = 0), or a plain
CSV with columns mass1, mass2, spin1x ... spin2z.
Output (./results): results_table.xlsx/.csv/.md in the same layout, all_runs.csv,
per-case diagnostic + histogram plots, summary plots.
