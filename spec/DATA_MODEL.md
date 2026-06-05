# Data Model And Metrics

## BBox Size Bins

BBox size is normalized as:

```text
bbox_area_pct = bbox_width * bbox_height / (frame_width * frame_height) * 100
```

Current matrix bins:

- `XXS`: `<0.15%`
- `XS`: `0.15-0.25%`
- `S`: `0.25-0.5%`
- `M`: `0.5-1%`
- `L`: `1-2%`
- `XL`: `2-4%`
- `XXL`: `4-8%`
- `XXXL`: `>=8%`

This makes reports comparable across different frame resolutions.

## BBox Quality Metrics

`quality_metrics(frame, bbox)` computes bbox-local image metrics:

- `laplacian_var` and `laplacian_score`: focus/edge high-frequency response.
- `tenengrad` and `tenengrad_score`: Sobel gradient energy.
- `edge_density` and `edge_density_score`: Canny edge coverage.
- `edge_contrast` and `edge_contrast_score`: contrast among detected edge pixels.
- `gradient_mean` and `gradient_score`: average gradient magnitude.
- `brightness`, `contrast`, `saturation`.
- `sharpness_score`: stricter composite contour score.
- `contour_clarity_score`: secondary contour quality score.
- `blur_score`: `100 - sharpness_score`.

## Frame Quality Metrics

`frame_quality_metrics()` computes whole-frame indicators:

- `frame_sharpness`
- `frame_brightness`
- `frame_exposure_score`
- `frame_motion_blur`
- `frame_contrast`
- `frame_lux_proxy`
- `frame_edge_density`
- `frame_quality_score`

`frame_lux_proxy` is relative and not calibrated physical lux.

## Perception Grade

`perception_grade(item)` uses a composite score. For video/webcam sources with tracking:

```text
score = 40% * confidence
      + 30% * strict bbox sharpness_score
      + 20% * track_stability_score
      + 10% * bbox frame-area percent score
```

For image batches without temporal tracking:

```text
score = 45% * confidence
      + 40% * strict bbox sharpness_score
      + 15% * bbox frame-area percent score
```

Grade thresholds:

- `A`: score >= 75
- `B`: score >= 60
- `C`: score >= 45
- `D`: score < 45

The bbox area score saturates at about `2%` of frame area. It is now a supporting term rather than the main quality factor.

## Tracking Stability Metrics

`track_stability_score` is only available for continuous video/webcam sources. It combines:

- `track_age_score`: how long the same track ID has survived.
- `track_continuity_score`: frame gap continuity within the track history.
- `track_smoothness_score`: center-point velocity/acceleration smoothness.
- `track_center_jump_pct`: latest normalized center jump as a frame-scale percent.

If the object ID drops and a new ID is created, the new track has low age. If the center path suddenly bends or jumps, smoothness is reduced. This makes tracking behavior visible in per-bbox Grade, modal analysis, product spec, and radar summaries.

For matrix-bin inspection, each retained sample may include the latest center-point trail for that ID. The frontend overlays this trail so users can visually inspect whether the bbox moved smoothly or jumped.

## Runtime Sample Cache

Intermediate image samples are intentionally bounded:

- classic examples are capped by `MAX_EXAMPLES`
- per-matrix-bin retained samples are capped by `MAX_BIN_SAMPLES`
- `/api/start` calls `reset_job()`, clearing examples, bin samples, quality history, track history, and current-object state before the next run

This keeps repeated runs from accumulating unnecessary in-memory image data. Exported reports remain persistent output files under `reports/` and the configured Downloads mount.

## Classic Samples

Classic samples are representative quality examples collected during `update_stats()`:

- Low contour clarity: `sharpness_score < 35`
- Small target: `bbox_area_pct < 0.5`
- Recommended quality: `conf >= 0.70 and sharpness_score >= 55`

They are not per-ID stability samples. Track stability is described by product spec indicators.

## Product Spec Indicators

`product_spec_indicators()` summarizes customer-facing operating conditions:

- `stable_track_count`: tracks with at least 5 sampled bbox hits.
- `stable_bbox_sample_count`: bbox samples belonging to stable tracks.
- `tracking_stability_score`: stable samples divided by all bbox samples.
- `max_simultaneous_bbox` and `avg_simultaneous_bbox`: model load/capacity view.
- `recommended_min_bbox_area_pct`: 10th percentile bbox ratio among usable stable samples.
- `avg_frame_lux_proxy`, `avg_frame_exposure_score`, and `dynamic_clarity_score`: environmental quality indicators.
