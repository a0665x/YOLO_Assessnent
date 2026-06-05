# UI

## Layout

The dashboard is a dark operational interface with:

- left control sidebar for model/source/report controls
- top metrics strip
- live inference stream
- bbox frame-ratio/confidence matrix
- live A/B/C/D grade distribution chart
- live metric distribution chart
- image quality radar and model perception radar
- current bbox table
- classic sample gallery
- bbox analysis modal

Two-column analytic rows are resizable on desktop:

- matrix/radar row
- grade-distribution/metric-distribution row

`static/app.js` inserts a `split-resizer` handle for each `.resizable-row`, stores the selected ratio in `localStorage`, and calls `resizeAllCanvases()` plus chart redraws after pointer movement. Canvas backing dimensions must follow the rendered column width; do not rely only on CSS scaling because it compresses chart labels and plots.

## Localization

The UI supports `zh`, `en`, `ja`, and `ko` through the `I18N` object in `static/app.js`. Reports receive the selected language through `/api/report`.

## Matrix Interaction

The matrix x-axis uses bbox frame area percent bins plus a size-class label:

```text
XXS <0.15%, XS 0.15-0.25%, S 0.25-0.5%, M 0.5-1%,
L 1-2%, XL 2-4%, XXL 4-8%, XXXL >=8%
```

Each drawn matrix cell records canvas coordinates and its `(area_bin, conf_bin)` key.

When a user clicks a populated cell:

1. UI calls `/api/bin_sample`.
2. A random retained original-frame sample appears below the matrix.
3. The original detected bbox is drawn as a solid red reference box with `alpha=0.5` red fill.
4. If the sample has tracking history, a cyan center-point trail is drawn behind the bbox so users can inspect ID path stability.
5. The user can drag a rectangle on the image.
6. During drag and on mouse release, UI displays:
   - drawn bbox area percent of the full frame
   - estimated pixel width and height

The reset button clears only the user-drawn rectangle, not the reference sample bbox.

## Hover Explanations

Each table grade chip exposes the current A/B/C/D formula on hover/focus:

```text
Video/Webcam: Grade = 40% Conf + 30% clarity + 20% tracking stability + 10% bbox size.
Images: Grade = 45% Conf + 40% clarity + 15% bbox size.
A>=75, B>=60, C>=45, D<45.
```

The live A/B/C/D bar chart also exposes grade definitions on hover. The metric distribution chart exposes concise metric definitions and whether high values are desirable.

## BBox Modal Metric Hierarchy

The sample modal separates metrics into three layers:

1. `Grade Drivers`: the final score inputs: confidence, bbox sharpness, tracking stability, and bbox frame-area percent.
2. `BBox Image Processing Details`: diagnostic sharpness/edge/contrast signals that explain why the bbox clarity score is high or low.
3. `Tracking And Frame Context`: track smoothness/continuity plus whole-frame quality such as exposure, blur, and lux proxy.

If many image-processing values are `0`, the UI explains likely causes: empty crop, missing debug metrics, old cached sample, or an inapplicable source type.

## Report Download

`POST /api/report` creates the report, and the frontend immediately requests `/reports/<file>?download=1` through a hidden anchor. This triggers the browser-native download flow, so Chrome shows it in the download list. The same report is still retained under `reports/` and the Docker-mounted Downloads folder.

## Classic Samples

Classic Samples are quality archetypes, not tracking-stability samples. They explain what kinds of bbox conditions the system observed:

- low contour clarity
- small target
- recommended quality

The user should use product spec/tracking indicators for ID stability and stable bbox capacity.
