# AI-Powered Digital Twin of India’s Climate — Data, Models, and Design

## Overview

To build a Digital Twin of India’s climate, we first catalog relevant national datasets and integrate them with AI-driven modeling. Key national sources (IMD, ISRO) provide long-term, high-resolution observations that enable training and validating AI models. The twin combines gauge-based grids, satellite archives, and reanalysis to form an integrated dataset for modeling and forecasting.

---

## 1. Key Data Sources & Parameters Catalog

A robust climate digital twin depends on sovereign data streams and standardized physical bounds. The table below outlines the core datasets utilized in the Indian Climate Digital Twin (ICDT) architecture.

### 1.1 Dataset Catalog

| Dataset Group | Specific Source / Platform | Parameter / Variable | Spatial Resolution | Temporal Resolution | Temporal Coverage | Native Format | Metadata & API Access |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **IMD Gridded Rainfall** | IMD Pune | Daily precipitation (gridded) | 0.25° × 0.25° (≈ 27 km) | Daily | 1901 – 2024 (124 yrs) | NetCDF / ASCII | Derived from 6,955 stations via modified Shepard's interpolation (Shepard, 1968) |
| **IMD Gridded Temp** | IMD Pune | Daily Max / Min Temperature | 1° × 1° (≈ 110 km) | Daily | 1951 – 2024 (73 yrs) | NetCDF / ASCII | Interpolated using thin-plate spline algorithms, calibrated against local stations |
| **ISRO MOSDAC Imager** | INSAT-3D / 3DR | Outgoing Longwave Radiation (OLR), Sea Surface Temp (SST), Land Surface Temp (LST) | 4 km (IR) / 1 km (Vis) | 15 minutes | 2013 – Present | HDF5 (Level-2 / L2B) | API access via MOSDAC; contains geo-location and calibration coefficients |
| **ISRO MOSDAC Rain** | INSAT-3D / 3DR | Quantitative Precipitation Estimation (QPE) | 4 km | 15 minutes | 2013 – Present | HDF5 (L2B / L3) | Derived using HEM (Hydro Estimator Method), IMSRA, and GPI techniques |
| **ISRO MOSDAC Sounder** | INSAT-3D / 3DR | Atmospheric temperature profiles, Water vapor profiles, Ozone profiles | 10 km × 10 km | Hourly | 2013 – Present | HDF5 (L2B) | 40 pressure levels (1000 hPa to 0.1 hPa); includes Lifted Index, K-Index, Total Precipitable Water |
| **ISRO Bhuvan (NRSC)** | Bhuvan Geoportal | Land Use / Land Cover (LULC), Glacier inventory, Water bodies extent | 56 m (AWiFS) / 24 m (LISS-III) | Annual / Seasonal | 1972 – Present | GeoTIFF / WMS | Managed by NRSC; critical for soil-moisture and surface albedo parameterization |
| **ISRO NICES ECVs** | NICES Portal | Surface Soil Moisture, Snow Cover Area, Active Fire Maps, Chlorophyll-a | 0.05° – 0.25° | Daily / 8-day composite | 2011 – Present | NetCDF / GeoTIFF | National Information System for Climate and Environment Studies (NICES) |
| **Reanalysis (Global)** | ECMWF ERA5 | u/v winds, humidity, geopotential, soil temperature, pressure | 0.25° × 0.25° (37 levels) | Hourly | 1940 – Present (5d lag) | NetCDF / GRIB | CDS API access; boundary conditions for regional downscaling |
| **Reanalysis (Regional)** | NCMRWF IMDAA | Wind fields, soil moisture, surface flux, heat flux | 0.12° × 0.12° (≈ 12 km) | 3-hourly | 1979 – 2020 | GRIB2 | Regional reanalysis customized for the Indian monsoon region |

### 1.2 Portal Interfaces
1. **MOSDAC (Meteorological & Oceanographic Satellite Data Archival Centre):** Serves as the primary real-time stream for INSAT-3D/3DR meteorological parameters, providing atmospheric profiles, cloud motion vectors (CMV), fog, and aerosol optical depth (AOD).
2. **Bhuvan Geoportal:** Delivers foundational GIS layers, Digital Elevation Models (Cartodem 1-arc second), Land Use/Land Cover changes, and near-real-time disaster monitoring data (flood inundation, landslide inventories, active forest fires via MODIS/VIIRS).
3. **NICES Portal:** Distributes Essential Climate Variables (ECVs) categorized across atmospheric (cloud top temperature, lightning), terrestrial (soil moisture, soil carbon), and oceanographic (ocean surface current, wind stress) domains.

---

## 2. Data Preprocessing, Quality Control, and Multi-Source Fusion

Raw observational data from stations and satellites exhibits discrepancies in projection, missing data, and systematic sensor biases. The ICDT preprocessing pipeline operates in four stages to generate high-resolution analysis fields.

```
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ IMD Grid     │   │ INSAT HDF5   │   │ Bhuvan GIS   │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         ▼                  ▼                  ▼
  ┌────────────────────────────────────────────────────┐
  │ 1. Data Cleaning & QC (Range, Spatiotemporal)      │
  └────────────────────────┬───────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────────┐
  │ 2. Spatial Alignment & Re-gridding (xarray, GDAL)  │
  └────────────────────────┬───────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────────┐
  │ 3. Bias Correction & Fusion (Quantile Mapping, DA) │
  └────────────────────────┬───────────────────────────┘
                           ▼
  ┌────────────────────────────────────────────────────┐
  │ 4. Derived Features & Zarr Serialization (Dask)   │
  └────────────────────────────────────────────────────┘
```

### 2.1 Quality Control (QC) Protocols
- **Range Verification:** Sensor values are checked against absolute physical limits (e.g., negative rainfall is zeroed; relative humidity must lie in $[0, 100]\%$; surface temperature must remain within $[-10, 60]^\circ\text{C}$).
- **Temporal Consistency:** Sudden jumps in hourly sensor readings (e.g., temperature changes $>15^\circ\text{C}$ in 1 hour) are flagged as sensor anomalies and imputed using temporal interpolation.
- **Spatial Consistency:** Neighborhood checks compare individual station or grid points with adjacent nodes. Spatial outliers are flagged and filled using **Kriging interpolation**.
- **Cross-Variable Sanity:** Relative humidity, air temperature, and dewpoint are evaluated jointly. Discrepancies that violate thermodynamic relations are corrected.

### 2.2 Spatial and Temporal Alignment
- **Re-gridding:** Satellite coordinates (geostationary projection for INSAT) and IMD ASCII grids are resampled to a common, high-resolution $0.05^\circ \times 0.05^\circ$ (≈ 5 km) spatial grid using **bilinear interpolation** for continuous variables (temperature, humidity) and **conservative remapping** (first-order) for discontinuous variables (precipitation) to preserve mass conservation.
- **Temporal Resampling:** INSAT-3D/3DR (15-min) and IMD AWS stations (hourly) are aggregated into hourly grids. Daily grids (like IMD gridded rain) are disaggregated to hourly fluxes using INSAT QPE temporal weights as a proxy.
- **Software Stack:** Built on `xarray` for multidimensional slicing, `dask` for out-of-core parallel execution, `Pyresample` for geographic projection transformations, and `rioxarray` for GeoTIFF-to-NetCDF mapping.

### 2.3 Multi-Source Data Fusion & Assimilation (DA)
To produce a continuous "best estimate" (analysis state) of the atmosphere, we implement a hybrid assimilation framework:
- **Baseline Fusion:** Weighted averages and multi-variable regression (e.g., merging satellite QPE with IMD gauge grids using rain-gauge densities as weights).
- **Advanced Data Assimilation:**
  - **Ensemble Kalman Filter (EnKF):** Updates the model state covariance dynamically using a 50-member ensemble.
  - **3DVar / 4D-Var:** Formulates data assimilation as a variational optimization problem.
  - **Machine Learning Data Assimilation (ML-DA):** A convolutional U-Net surrogate model (**UNetKF**) is trained to emulate Kalman filter updates. This bypasses the expensive matrix inversions of EnKF, delivering a 100× speedup in generating initial analysis states.
  - **FuXi-DA / learned DA:** End-to-end differentiable data assimilation that enforces physical boundary conditions using neural operators.

---

## 3. AI/ML Modeling for Climate Variables and Extremes

No single model captures the spectrum of climate scales in India. The ICDT utilizes a hybrid ensemble architecture designed to balance local convective details and seasonal teleconnections.

### 3.1 Candidate Model Classes

```
                            ┌────────────────────────┐
                            │    Assimilated State   │
                            └───────────┬────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
  │      ConvLSTM       │    │  Swin Transformer   │    │  Fourier Operator   │
  │ (Short-Range Rain)  │    │  (Medium-Range T)   │    │ (Physically Stable) │
  └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘
             │                          │                          │
             └──────────────────────────┼──────────────────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │    Stacking / Gating   │
                            └───────────┬────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │   Physics Loss (PINN)  │
                            └───────────┬────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │  Inference / Downscale │
                            └────────────────────────┘
```

#### 1. ConvLSTM / ConvGRU (Spatiotemporal RNNs)
- **Best For:** Short-range (0–24 hours) convective rainfall nowcasting and storm tracking.
- **Mechanism:** Replaces the fully connected operations in standard LSTMs with convolutional layers, capturing spatial structure and temporal dynamics simultaneously.
- **Skill Score (India):** ConvLSTM models achieve a Root Mean Squared Error (RMSE) of **0.21 mm/day** in Bengaluru, **0.52 mm/day** in Mumbai, and **0.48 mm/day** in Delhi for 24-hour lead times.

#### 2. Swin Transformer (Hierarchical Vision Transformer)
- **Best For:** Medium-range (1–10 days) regional forecasting of temperature, winds, and pressure.
- **Mechanism:** Utilizes shifted window self-attention to scale self-attention computations linearly with image size. Captures long-range teleconnections (e.g., El Niño-Southern Oscillation, Indian Ocean Dipole influence on monsoon onset).
- **Limitation:** High GPU memory consumption; requires patch-based tokenization.

#### 3. Fourier Neural Operator (FNO)
- **Best For:** Solving partial differential equations (PDEs) governing atmospheric dynamics (Navier-Stokes equations).
- **Mechanism:** Parameterizes the integral kernel in Fourier space, learning operators that map initial atmospheric states to future states independent of grid resolution (zero-shot super-resolution).

#### 4. Physics-Informed Neural Networks (PINNs) & ClimODE
- **Best For:** Enforcing physical consistency (conservation of mass, momentum, and energy) over long-term projections.
- **Mechanism:** Incorporates physical residuals (e.g., wind divergence, pressure gradients, boundary conditions) directly into the loss function:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1 \mathcal{L}_{\text{mass}} + \lambda_2 \mathcal{L}_{\text{momentum}}$$
- **ClimODE Integration:** Integrates neural Ordinary Differential Equations (ODEs) with value-conserving continuous dynamics, ensuring that moisture and heat do not accumulate or disappear artificially during simulation rollouts.

### 3.2 Training and Validation Strategy
- **Temporal Splitting:** Training data spans 1951–2018; validation covers 2019–2021; testing is reserved for the 2022–2024 monsoons (out-of-distribution evaluation).
- **Loss Functions:** 
  - For temperature: Mean Squared Error (MSE) + Mean Absolute Error (MAE) loss.
  - For precipitation: **Weighted Extreme Loss** to penalize under-prediction of heavy monsoon rain ($>100$ mm/day):
    $$\mathcal{L}_{\text{rain}} = \text{MSE} + \gamma \cdot \mathbb{I}(y_{\text{obs}} > \tau) \cdot (y_{\text{pred}} - y_{\text{obs}})^2$$
- **Explainable AI (XAI):**
  - **Grad-CAM:** Visualizes which satellite bands and regions dominate convective rain predictions.
  - **SHAP (SHapley Additive exPlanations):** Details the contribution of individual meteorological features (e.g., surface temperature, specific humidity at 850 hPa) to forecast anomalies.

---

## 4. Digital Twin PoC Architecture & Workflow

A Proof-of-Concept (PoC) of the ICDT is designed for the state of **Maharashtra** as a representative pilot region, spanning the Western Ghats (orographic rainfall), coastal Mumbai (urban flood risk), and the Deccan plateau (drought-prone agriculture).

### 4.1 Step-by-Step Pilot Pipeline

1. **Step 1: Pilot Basin Selection**
   - Region: Godavari River Basin / Mumbai Metropolitan Region ($15^\circ\text{N} - 21^\circ\text{N}, 72^\circ\text{E} - 80^\circ\text{E}$).
   - Variables: Gridded precipitation, land surface temperature, soil moisture, runoff.
2. **Step 2: Automated Ingestion Pipeline**
   - Ingest IMD gridded daily rainfall and temperature records.
   - Fetch real-time INSAT-3D L2B products from MOSDAC APIs every 15 minutes.
   - Load Bhuvan LULC maps and soil characteristics to initialize land surface variables.
3. **Step 3: State Estimation (Data Assimilation)**
   - Fuse INSAT QPE, AWS gauge readings, and ERA5 variables using a localized Ensemble Kalman Filter.
   - Generate hourly 5 km analysis fields for precipitation and temperature.
4. **Step 4: AI Model Execution**
   - Run the trained ConvLSTM-GNN ensemble on NVIDIA GPU instances.
   - Output 1-day, 3-day, and 7-day probabilistic forecasts with uncertainty bounds.
5. **Step 5: Scenario Simulation ("What-If" Sandbox)**
   - Run counterfactual simulations.
   - *Example Scenario:* Shift LULC by converting 20% agricultural land to urban build-up in the Pune-Mumbai corridor.
   - *Example Perturbation:* Inject a $+2.5^\circ\text{C}$ regional warming scenario (RCP 8.5 target for 2050) and evaluate changes in extreme rainfall frequency.
6. **Step 6: Dashboard and Decision Support**
   - Visualize simulated outputs on a CesiumJS 3D globe.
   - Provide district-level crop water requirement charts and flood hazard overlays for urban planners.

---

## 5. Evaluation and Validation Metrics

To ensure the digital twin is statistically sound and physically realistic, we validate predictions against withheld ground-truth records using a dual evaluation framework.

### 5.1 Statistical Metrics
- **Deterministic Forecasts:** 
  - Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
  - **Threat Score (Critical Success Index - CSI):** Evaluates binary precipitation events (rain/no-rain) at various thresholds ($5$ mm, $20$ mm, $50$ mm, $100$ mm):
    $$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$$
  - **Nash-Sutcliffe Efficiency (NSE):** Validates streamflow forecasts in the hydrological module.
- **Probabilistic Forecasts:**
  - **Continuous Ranked Probability Score (CRPS):** Evaluates the calibration and sharpness of the 50-member ensemble.
  - **Brier Score:** Validates probability forecasts for extreme thresholds (e.g., probability of temperature exceeding $45^\circ\text{C}$).

### 5.2 Physical Consistency Audits
- **Water Balance:** Ensures that total precipitation matches the sum of evapotranspiration, runoff, and soil moisture storage change over the pilot basin.
- **Geostrophic Balance:** Verifies that wind fields generated by the model correspond to pressure fields, preventing unphysical shear gradients.
- **Monsoon Index Alignment:** Validates model representation of the monsoon trough, the Low-Level Jet (LLJ), and the Tibetan Anticyclone.

---

## Citations

- Pai, D. S., et al. (2014). "Development of a new high spatial resolution (0.25° × 0.25°) long period daily gridded rainfall dataset over India." *IMD Meteorological Monograph*.
- Shi, X., et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting." *Advances in Neural Information Processing Systems (NeurIPS)*.
- Lam, R., et al. (2023). "Learning skillful medium-range global weather forecasting (GraphCast)." *Science*.
- Bonev, B., et al. (2023). "ClimODE: Climate- and Weather-Informed Neural ODE for Modeling Atmospheric Dynamics." *International Conference on Machine Learning (ICML)*.
- Pathak, J., et al. (2024). "FourCastNet 3: Geometrically-constrained machine learning for atmospheric simulation." *arXiv:2406.04182*.
