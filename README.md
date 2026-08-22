[English](README.md) | [日本語](README.ja.md)

# matuge-change

**Keep the product. Change only the person.**

Matsuge Change is a proof-of-concept web application for e-commerce sellers. It transforms the person in
photos and videos into an AI model while preserving the appearance of the false-eyelash product they are
actually wearing as faithfully as possible.

- [Demo video (YouTube)](https://www.youtube.com/watch?v=hN2X6LOEeXA)
- [Live demo](https://matuge-change.onrender.com/)
- [Project background (Japanese)](docs/project-background.md)

## The problem

The appearance of false eyelashes depends on their length, clusters, spacing, curl, outer-corner spread,
and placement on the eye. An e-commerce image must show the product actually being sold, not merely a
similar-looking alternative.

However, when generative AI changes the person wearing the product, it may regenerate the eye area as well
and replace the product lashes with a different design.

![Generative AI changes the product lashes while changing the person](img/problem-generative-ai-changes-product.png)

Matsuge Change therefore separates the product region from the generative-AI process instead of asking the
model to reproduce the product correctly.

```text
Source image / video
   │
   ├─ Product region ─────────────┐
   │      extract / preserve      │
   │                              │
   └─ Person region               │
          ↓                       │
       Edit with generative AI    │
          ↓                       │
          └───────────────────→ Recompose
```

Asking a general-purpose image generation model to cut out the lashes directly produced a regenerated
product rather than a segmentation result. The actual output and failure analysis are documented in
[Failure case: asking generative AI to perform the cut-out (Japanese)](docs/generative-ai-cutout-failure.md).

## How it works

Still images and videos use different preservation strategies because video must remain temporally stable
and follow blinking.

| Mode | Product-preservation method | Output |
| --- | --- | --- |
| Still image | Extract an alpha matte from the eye ROI, then place it on the AI-edited image using only a similarity transform and alpha compositing | PNG |
| Video | Preserve each source frame's eye region and align the AI-edited image to the tracked face | MP4 |

### Still images

The user supplies a photo with the product attached and an AI-edited photo. Face landmarks, evidence maps,
a trimap, and closed-form alpha matting produce the Product RGBA layer. A three-state brush—product,
unknown, and background—can correct automatic extraction. Side profiles and close-ups can use manual ROIs.

```text
Product photo → Eye ROI → Evidence → Trimap → Alpha Matting
                                                  ↓
AI-edited photo ← Similarity transform + alpha ← Product RGBA
```

![Generate a trimap and extract the product lashes](img/static-1-input-trimap-extract.png)

![Recompose the extracted product lashes onto the AI-edited image](img/static-2-fitting-recompose.png)

Details: [Still-image algorithm (Japanese)](docs/static-image-algorithm.md) /
[Simple-mode specification (Japanese)](docs/static-image-simple-mode.md)

### Video

The system selects the sharpest frame with the most open eyes. That one frame is edited with an external
AI service. For every output frame, the AI-edited image follows the tracked face while the original eye
region—including lashes, eyelids, and blinking—is composited back. Independent per-frame alpha extraction
is deliberately avoided because it flickers.

```text
Source video → Select best frame → Edit person with external AI
      ↓                                ↓
Eye region from each frame ─────→ Face tracking + composite → MP4
```

Details: [Video algorithm (Japanese)](docs/video-algorithm.md) /
[Video design decisions (Japanese)](docs/video-approach.md)

## Product-preservation definition

In this project, **Generative Product Preserve** means that generative AI does not redraw the false-eyelash
product. Instead, the system reuses the product appearance derived from the real source image as faithfully
as practical.

- Still-image product layers may only undergo a similarity transform: translation, uniform scaling,
  rotation, and reflection
- Free-form deformation, perspective transforms, non-similarity warps, and generative product completion
  are not allowed
- In video, the source eye region containing the product is not geometrically transformed; only the
  AI-edited image is warped
- MediaPipe Face Landmarker is used to detect the face and eyes
- Still images pass through foreground estimation and interpolation; videos pass through boundary
  feathering and H.264 re-encoding
- The final output is therefore not guaranteed to match the source image pixel for pixel

See [Design philosophy (Japanese)](docs/design-philosophy.md) for the precise guarantees and rationale.

## Current capabilities

- Simple still-image workflow from the product photo and AI-edited photo through extraction and recomposition
- Automatic landmark-based ROIs and manual ROIs for side profiles and close-ups
- Diagnostic layers including Probability, Trimap, Alpha, and Product RGBA
- Three-state brush, undo/redo, thresholds, and similarity-transform adjustments
- Persistent sessions and processing history
- Catalog registration and shape-similarity search for extracted product lashes
- Video best-frame selection, face tracking, eye-region preservation, and MP4 output
- Local persistence with optional Supabase integration
- Regression measurement with a Synthetic Benchmark

The PoC deliberately targets [one false-eyelash product currently sold on Amazon Japan](https://www.amazon.co.jp/dp/B0GFJSHBWT).

## Quick start

Docker or WSL2 is recommended. This path does not require a host Python environment or a manually installed
MediaPipe model.

```bash
git clone https://github.com/okajun35/matuge-change.git
cd matuge-change
docker compose up --build
```

Open <http://localhost:8000> after `docker compose ps` reports `healthy`. The initial dependency download and
MediaPipe/numba startup may take some time.

| URL | Page |
| --- | --- |
| `/` | Product catalog |
| `/extract.html` | Still-image mode |
| `/video.html` | Video mode |

Extraction results and catalog data remain in `./data` on the host. See
[Development environment and quality checks (Japanese)](docs/development.md) for local Python setup and
testing instructions.

## Basic usage

### Still images

For normal cases, use the simple workflow at `/extract.html`.

1. Select the product photo, in which the person is wearing the lashes, and the AI-edited photo
2. Select **Run processing** to perform analysis, matting, and recomposition
3. Compare the result with the original and save the completed PNG
4. If extraction or placement is insufficient, open the detailed controls and adjust the ROIs, three-state
   brush, thresholds, or alignment

A photo without the product is optional and available in the detailed controls. For side profiles and eye
close-ups that automatic face detection cannot handle, manually specify ROI-A for extraction and ROI-B for
placement.

### Video

1. Analyze a video of a person wearing the product at `/video.html`
2. Save the selected best frame
3. Edit the person with an external AI service, instructing it not to alter the eye area
4. Upload the AI-edited image and run video compositing
5. Review the preview and save the MP4

## Quality evaluation

The `evaluation/` directory contains a Synthetic Benchmark that runs production code against generated data
with known ground-truth masks and alpha. Results must be interpreted in three layers:

- **A**: Properties of the code path that also apply to real images
- **B**: Relative trends and robustness across conditions
- **C**: Absolute scores on synthetic data, used only for regression detection—not as real-image accuracy

See [evaluation/README.md](evaluation/README.md) for the procedure, metrics, and limitations of synthetic data,
and [docs/benchmark-findings.md](docs/benchmark-findings.md) for known issues in the current implementation.

## Development

```bash
. .venv/bin/activate
python -m pytest
ruff check
pre-commit run --all-files
```

Features and bug fixes follow test-driven development: add a failing test first, implement the minimum fix,
then refactor while keeping the test green. See [AGENTS.md](AGENTS.md) for repository-specific rules and
[docs/development.md](docs/development.md) for environment setup and commands.

## Documentation

The complete documentation index is in [docs/README.md](docs/README.md). Key documents:

- [Project background (Japanese)](docs/project-background.md)
- [Design philosophy (Japanese)](docs/design-philosophy.md)
- [Still-image algorithm (Japanese)](docs/static-image-algorithm.md)
- [Video algorithm (Japanese)](docs/video-algorithm.md)
- [Deployment and operations (Japanese)](docs/deployment.md)
- [Engineering handover notes (Japanese)](docs/handover.md)
