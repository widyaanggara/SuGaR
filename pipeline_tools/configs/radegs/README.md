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

## Shared comparison protocol

New jobs now pass `--visual_protocol shared`, with `--image_downscale` and
`--image_max_size` from the same dashboard settings as 3DGS/SuGaR (defaults 1/1600).
The launcher overrides the native camera resize for training and rendering with
the shared longest-side rule, including portrait images and no upscaling. This
overrides the YAML's 1920-width request; allocator GC and other method parameters
remain active. Mesh extraction retains the method's native settings.

Metrics use the common PSNR/SSIM implementation and VGG LPIPS with [-1,1] inputs.
SuGaR/3DGS metrics are now calculated on 8-bit RGB values, matching the PNG inputs
used by external evaluators. Protocol metadata invalidates old evaluation caches.
Existing checkpoints can be evaluated at the shared resolution without retraining;
this does not retroactively make their training resolution equal. New training
cannot resume a checkpoint with a different recorded protocol.

The profiles do not guarantee a 30-minute run. Verify throughput on the server,
including after regularization activates at iteration 7000.

## Dashboard mesh backend

Dashboard RaDe-GS jobs and full pipelines default to `scalable`, including queued
jobs without a saved backend setting. The batch runner passes `--mesh_backend
scalable` only to RaDe-GS postprocessing. This selects `mesh_extract_tnt.py`
(ScalableTSDFVolume), avoiding the VoxelBlockGrid extraction that crashed on the
server. Training settings remain unchanged. Restart the dashboard after updating
the code; already running subprocesses keep their original command.

The dashboard API accepts `radegs_mesh_backend: native` for explicit comparisons;
the batch CLI equivalent is `--radegs_mesh_backend native`. The standalone
`run_external_method.py` retains its native default unless `--mesh_backend` is
specified. Backend metadata and existing mesh backups are managed by that runner.
Scalable meshes can differ from native meshes and require geometry evaluation.
Success on one statue does not guarantee every dataset or environment will succeed.

The scalable script runs through `pipeline_tools/radegs_mesh.py`, which preserves
its camera/render/TSDF parameters and guards postprocessing when fewer than eight
components exist. It rejects empty/tiny-only meshes and reads both output PLYs
back to check vertices and triangles. Resume also checks that the mesh receipt
refers to the current checkpoint before skipping completed work.
