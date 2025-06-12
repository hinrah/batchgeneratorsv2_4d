# Changelog

## 4D (3D + time) support for spatial augmentation

This release teaches `SpatialTransform` to handle 5D tensors of shape
`(c, t, x, y, z)` in addition to the existing 4D `(c, x, y, z)` layout, and adds two
small wrapper transforms for composing time-aware augmentation pipelines.

### Added

- **`TransformEachTimestep`** (`transforms/utils/random.py`): wraps an arbitrary
  transform and applies it independently to every timestep of a 5D input.
  `get_parameters()` asserts 5D input and returns `num_timesteps` from `shape[1]`;
  `apply()` slices `data_dict[key][:, t]`, runs the wrapped transform, and writes the
  result back in place. Enables reuse of all existing 3D transforms (noise, blur,
  intensity, …) on time-resolved data.
- **`TransformNothing`** (`transforms/utils/doNothing.py`): a pass-through wrapper
  that forwards `data_dict` to the wrapped transform unchanged. Serves as the
  counterpart to `TransformEachTimestep`, so callers can pick a time-handling wrapper
  based on dimensionality without branching around every transform in the pipeline.

### Changed

**`SpatialTransform` — time-dimension awareness**
- New attribute `self.spatial_patch_size = patch_size[-3:]`. The identity grid,
  elastic offsets and deformation sigmas are now derived from the *spatial* part of
  the patch size only, so no geometric transformation is applied along time.
- `get_parameters()` now computes `spacial_dim = min(image.ndim - 1, 3)` instead of
  `image.ndim - 1`. Angles, scales, the affine matrix and deformation scales are
  therefore always built for at most three axes, and 5D input yields a 3D affine
  matrix rather than hitting the `Unsupported dimension` error.
- `_apply_to_image()`:
  - detects a time axis via `has_time_dim = img.ndim == 5`;
  - truncates `new_center` to its last three entries (spatial center only);
  - broadcasts one spatial grid across the leading axis with
    `torch.stack([grid] * img.shape[0], dim=0)` and performs a single `grid_sample`
    call for all timesteps;
  - adds/strips the leading dummy axis for 4D input so previous behaviour is preserved.
- `_apply_to_segmentation()`: same detection, center truncation and grid broadcast.
  The `nearest` path resamples all timesteps in one call; the `bg_style_seg_sampling`
  and multi-label fallback paths were reindexed from `[0][0]` to `[0]` with an extra
  unwrap when there is no time dimension.
- Cosmetic: whitespace/indentation cleanups, comment reflow, trailing whitespace removal.

### Compatibility

- Fully backwards compatible for 4D `(c, x, y, z)` input: the new code paths are
  gated on `ndim == 5`.
- For 4D use, `patch_size` semantics are unchanged. For 4D-with-time use, `patch_size`
  is expected to carry the time extent as its first entry; only the last three entries
  drive the spatial sampling grid.