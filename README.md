# Radiative Cooling API

FastAPI backend for spectrum-based radiative cooling and heating calculations.

## Render deployment

This repository is ready for Render. Select **Free** and deploy from the `main` branch. The `render.yaml` file sets the build, start, and health-check commands automatically.

After deployment, open `/health`. Expected JSON:

```json
{"status":"ok","service":"radiative-cooling-api"}
```

The calculation endpoint is `POST /api/v1/calculate`.

## Important model note

This first backend implements spectral interpolation, opaque-sample emissivity conversion (`ε = 1 − R − T`), solar weighting, Planck weighting, radiative power terms, convection, and equilibrium temperature. Its default solar and clear-sky spectra are transparent fallback models; replace them with validated AM1.5 and atmospheric datasets before publishing scientific values.
