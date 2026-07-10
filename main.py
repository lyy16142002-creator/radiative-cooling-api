from __future__ import annotations

from math import exp, pi
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

H = 6.62607015e-34
C = 299792458.0
KB = 1.380649e-23


class SpectrumPoint(BaseModel):
    wavelength: float = Field(gt=0, description="Wavelength in micrometres")
    value: float = Field(ge=0, le=1, description="Spectral property in 0–1")


class CalculationRequest(BaseModel):
    reflectance: list[SpectrumPoint] = Field(min_length=2)
    emissivity: list[SpectrumPoint] | None = None
    transmittance: list[SpectrumPoint] | None = None
    ambient_temperature: float = Field(default=300, ge=180, le=450)
    solar_irradiance: float = Field(default=900, ge=0, le=1400)
    convection_coefficient: float = Field(default=6, ge=0, le=100)
    mode: Literal["cooling", "heating"] = "cooling"

    @model_validator(mode="after")
    def ensure_spectrum(self):
        if max(point.wavelength for point in self.reflectance) < 2.5:
            raise ValueError("Reflectance data must include the solar band through at least 2.5 μm")
        return self


def _series(points: list[SpectrumPoint], wavelength_um: np.ndarray) -> np.ndarray:
    ordered = sorted(points, key=lambda point: point.wavelength)
    x = np.array([point.wavelength for point in ordered])
    y = np.array([point.value for point in ordered])
    return np.clip(np.interp(wavelength_um, x, y), 0, 1)


def _planck_exitance(wavelength_um: np.ndarray, temperature: float) -> np.ndarray:
    wavelength_m = wavelength_um * 1e-6
    exponent = H * C / (wavelength_m * KB * temperature)
    return 2 * pi * H * C**2 / (wavelength_m**5 * np.expm1(exponent))


def _clear_sky_tau(wavelength_um: np.ndarray) -> np.ndarray:
    return np.where((wavelength_um >= 8) & (wavelength_um <= 13), 0.8, np.where((wavelength_um >= 3) & (wavelength_um <= 5), 0.55, 0.08))


def _solve(request: CalculationRequest, film_temperature: float):
    solar_grid = np.linspace(0.3, 2.5, 500)
    ir_grid = np.linspace(2.5, 30, 1200)
    rho_solar = _series(request.reflectance, solar_grid)
    solar_weight = _planck_exitance(solar_grid, 5778.0)
    r_sol = float(np.trapezoid(rho_solar * solar_weight, solar_grid) / np.trapezoid(solar_weight, solar_grid))

    reflectance_ir = _series(request.reflectance, ir_grid)
    transmittance_ir = _series(request.transmittance, ir_grid) if request.transmittance else np.zeros_like(ir_grid)
    emissivity_ir = _series(request.emissivity, ir_grid) if request.emissivity else np.clip(1 - reflectance_ir - transmittance_ir, 0, 1)
    p_rad = float(np.trapezoid(emissivity_ir * _planck_exitance(ir_grid, film_temperature), ir_grid) * 1e-6)
    p_atm = float(np.trapezoid(emissivity_ir * (1 - _clear_sky_tau(ir_grid)) * _planck_exitance(ir_grid, request.ambient_temperature), ir_grid) * 1e-6)
    q_solar = (1 - r_sol) * request.solar_irradiance
    q_conv = request.convection_coefficient * (request.ambient_temperature - film_temperature)
    cooling_net = p_rad - p_atm - q_solar - q_conv
    return r_sol, emissivity_ir, ir_grid, p_rad, p_atm, q_solar, q_conv, cooling_net


def calculate(request: CalculationRequest):
    _, _, _, p_rad, p_atm, q_solar, q_conv, cooling_net = _solve(request, request.ambient_temperature)
    window = np.linspace(8, 13, 250)
    reflectance = _series(request.reflectance, window)
    transmittance = _series(request.transmittance, window) if request.transmittance else np.zeros_like(window)
    epsilon_window = _series(request.emissivity, window) if request.emissivity else np.clip(1 - reflectance - transmittance, 0, 1)
    window_emissivity = float(np.trapezoid(epsilon_window, window) / 5)

    temperature_grid = np.linspace(max(180, request.ambient_temperature - 80), request.ambient_temperature + 80, 81)
    curve = []
    for temp in temperature_grid:
        net = _solve(request, float(temp))[-1]
        curve.append({"temperature": round(float(temp), 3), "net_power": round(float(-net if request.mode == "heating" else net), 5)})
    equilibrium = temperature_grid[0]
    for left, right in zip(curve, curve[1:]):
        if left["net_power"] * right["net_power"] <= 0:
            equilibrium = left["temperature"] + (right["temperature"] - left["temperature"]) * (-left["net_power"]) / (right["net_power"] - left["net_power"])
            break
    net_power = -cooling_net if request.mode == "heating" else cooling_net
    return {
        "mode": request.mode,
        "net_power": round(float(net_power), 5),
        "solar_weighted_reflectance": round(float(_solve(request, request.ambient_temperature)[0]), 6),
        "window_emissivity_8_13um": round(window_emissivity, 6),
        "equilibrium_temperature": round(float(equilibrium), 3),
        "components": {"surface_radiation": round(p_rad, 5), "atmospheric_back_radiation": round(p_atm, 5), "solar_absorption": round(q_solar, 5), "convection": round(q_conv, 5)},
        "curve": curve,
        "model": "clear-sky-default-v1",
        "warning": "Default solar weighting uses a normalized 5778 K solar blackbody shape and the atmosphere uses a clear-sky default. Replace these with measured AM1.5 and atmospheric spectra for publication-grade calculations.",
    }


app = FastAPI(title="Radiative Cooling API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["https://radiative-cooling-lab.zhangruifeng0823.chatgpt.site"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "radiative-cooling-api"}


@app.post("/api/v1/calculate")
def radiative_calculation(request: CalculationRequest):
    try:
        return calculate(request)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
