# RaDe-GS YAML profiles

These two files are copied from `external/RaDe-GS/configs/` so they travel with
the main SuGaR repository when pushed/pulled. They are the authoritative profiles
for new dashboard jobs, not live links to the external repository copies.

- Bali statue classes use `togog_bali.yaml`.
- `TanksAndTemples` uses `tanks_temples.yaml`.

The runner reads and validates the selected YAML in the RaDe-GS Python environment
using `research_pipeline.config` and builds native commands using
`research_pipeline.commands`. That environment needs PyYAML and the modified
RaDe-GS repository containing those modules and `--enable_allocator_gc` support.
Missing dependencies or unsupported arguments fail explicitly before training.

Training, rendering, visual metrics, mesh settings and the pre-mesh resource guard
come from YAML. TogogLab retains dataset/output routing, GPU serialization,
checkpoint recovery, logs, and its separate LiDAR geometry workflow. YAML dataset,
output and report paths from the donor machine do not redirect dashboard outputs.
The runner adds recovery checkpoints at 7000/15000/30000 iterations.

Effective settings are saved in `radegs_effective_config.json`; settings affecting
the run are recorded in `togoglab_run.json`. A change rejects resume instead of
mixing experiments. Back up old RaDe-GS output before explicitly retraining with
the new profile. Existing queue jobs without `radegs_profile` retain `legacy`
behavior; create a new job to use YAML. CLI defaults to YAML, with
`--radegs_profile legacy` available for old runs. The downscale control now applies
to 2DGS; YAML owns RaDe-GS resolution.

The profiles do not guarantee a 30-minute run. Verify throughput on the server,
including after regularization activates at iteration 7000.
