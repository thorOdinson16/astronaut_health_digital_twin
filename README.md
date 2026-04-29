# Astronaut Digital Twin

**A Service-Oriented Hybrid Digital Twin for Coupled Sleep-Fatigue and Space Motion Sickness**

> A physics-based physiological simulation system modelling the bidirectional coupling between sleep homeostasis and vestibular adaptation in long-duration spaceflight.

---

## Table of Contents

- [Overview](#overview)
- [Scientific Background](#scientific-background)
- [System Architecture](#system-architecture)
- [Physics Models](#physics-models)
- [Monte Carlo Engine](#monte-carlo-engine)
- [BioGears Integration](#biogears-integration)
- [API Reference](#api-reference)
- [Frontend Dashboard](#frontend-dashboard)
- [Installation](#installation)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Academic References](#academic-references)
- [Known Limitations](#known-limitations)

---

## Overview

The Astronaut Digital Twin is a hybrid physiological simulation system that addresses a specific gap in existing aerospace medicine tools: the lack of mechanistic coupling between sleep physiology and vestibular adaptation in microgravity.

Existing tools model these systems independently. This project implements, for the first time in a single computable system, the explicit state-dependent feedback loop between the two most validated mathematical models in the literature — the **Borbely two-process sleep model (1982)** and the **Oman vestibular mismatch model (1982)** — and quantifies the synergistic excess risk that emerges from their interaction.

**The core research contribution** is demonstrating that treating sleep deprivation and Space Motion Sickness as independent systems systematically underestimates mission risk. The system provides a counterfactual analysis framework to quantify exactly how much additional risk the coupling produces.

### Key Features

- Coupled ODE system integrating Borbely sleep homeostasis with Oman vestibular adaptation
- Sleep-pressure-gated vestibular adaptation rate — the novel coupling mechanism
- Counterfactual analysis quantifying synergistic excess risk from the coupling
- Monte Carlo simulation across inter-individual physiological variability distributions
- Multi-astronaut simulation support (up to 5 crew members per run)
- BioGears integration for high-fidelity cardiovascular responses during discrete events
- Chronic-stress BioGears trigger at fatigue threshold ≥ 6.0 (Samn-Perelli scale)
- Selective event enablement — motion sickness, sleep disruption, and EVA/exercise toggleable independently
- EVA/exercise stress event modelling with metabolic workload simulation
- Sensitivity analysis endpoint for heart rate and sleep quality profiling
- Real-time 3D dashboard with Interstellar-themed visualisation
- Run comparison panel for side-by-side analysis of multiple simulation runs
- Groq LLM (Llama 3.3 70B) and Anthropic Claude integration for natural-language risk explanation
- In-browser PDF report generation with flight surgeon signature block

---

## Scientific Background

### The Core Problem

In microgravity, two physiological problems interact in a self-reinforcing cycle:

1. **Sleep disruption**: The ISS orbital period of 90 minutes creates 16 sunrises and 16 sunsets per day, causing continuous circadian misalignment. Barger et al. (2014) documented that ISS crew sleep an average of 6.09 hours per night versus a 7.17-hour preflight baseline, with mean sleep efficiency of 0.71.

2. **Space Motion Sickness (SMS)**: Affects 67–75% of astronauts in the first 72 hours of microgravity. The vestibular system — specifically the otolith organs — receives a signal radically different from the Earth-calibrated expectation the brain has maintained across a lifetime. The resulting mismatch between expected and actual otolith signals drives nausea, spatial disorientation, and general malaise.

### Why Coupling Matters

The key insight this project is built around: **fatigue from sleep deprivation slows vestibular adaptation**. Vestibular adaptation is a neural plasticity process — the brain updating its internal model of expected otolith signals — and like all learning, it is impaired by sleep deprivation.

This creates a closed feedback loop:

```
High sleep pressure
    → Suppressed vestibular adaptation rate
        → Sustained vestibular mismatch
            → Motion sickness episodes
                → Disrupted sleep + physiological stress
                    → Higher sleep pressure
```

This loop cannot be captured by any pair of independent models. It is a system-level phenomenon.

---

## System Architecture

The system uses a three-layer architecture with strict separation between the physics core, discrete event management, and the analytics/API layer.

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND DASHBOARD                      │
│     Three.js 3D Scene · Chart.js · Groq AI · PDF Export     │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────┐
│                     FASTAPI BACKEND                          │
│   /api/simulation · /api/data · /api/health · /api/config   │
└─────┬──────────────────────────────────┬────────────────────┘
      │                                  │
┌─────▼──────────┐            ┌──────────▼──────────────────┐
│ ANALYTICS LAYER │            │     SIMULATION LAYER        │
│ Monte Carlo     │            │ EventScheduler              │
│ Risk Engine     │            │ MotionSicknessEvent         │
│ Trend Analysis  │            │ SleepDisruptionEvent        │
│ CouplingDiag.   │            │ ExerciseStressEvent         │
└─────────────────┘            └──────────┬──────────────────┘
                                          │
┌─────────────────────────────────────────▼──────────────────┐
│                      PHYSICS CORE                           │
│  BorbelyModel · VestibularMismatchModel · FatigueModel     │
│              PhysicsEngine · CouplingEngine                 │
└──────────────────────────────┬─────────────────────────────┘
                               │ on discrete events only
┌──────────────────────────────▼──────────────────────────────┐
│                    BIOGEARS ENGINE                           │
│   AcuteStressData · ExerciseData · SleepData + PVT         │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Physics Core** (`core/`) — The mathematical heart of the system. Three classes — `BorbelyModel`, `VestibularMismatchModel`, `FatigueModel` — are composed by `PhysicsEngine`, which is called once per timestep. The calling sequence enforces data dependencies: Borbely first (produces `S_norm` and sleep quality), then Vestibular (consumes `S_norm`), then Fatigue (consumes both). `CouplingEngine` runs in parallel, accumulating per-step diagnostics — k-suppression, joint-risk windows, and the coupling probability traces that feed `CouplingDiagnostics`.

**Simulation and Events Layer** (`events/`) — Manages the discrete stochastic events that punctuate the continuous ODE simulation. `EventScheduler` maintains a priority queue of pending events and applies active event effects at each timestep. The three event types each implement `sample_onset()`, `get_duration()`, `apply_effect()`, and `get_biogears_perturbation()`. Event types are independently toggleable at simulation launch via `SimulationConfig.enable_*` flags, which are wired directly to the scheduler's disabled-event list.

**Analytics and API Layer** (`analytics/`, `api/`) — Sits above the simulation and provides outputs for consumption. The risk engine computes threshold-crossing metrics; the Monte Carlo engine runs N parallel trajectories with sampled biological variability; `CouplingDiagnostics` performs the counterfactual analysis that is the paper's primary result. A `sensitivity` endpoint allows exploration of heart rate and sleep quality profiles across physiological parameter ranges.

---

## Physics Models

### Borbely Two-Process Sleep Model

Implemented in `core/fatigue_model.py` as `BorbelyModel`. Euler integration at configurable timesteps (default 5 minutes for main simulation; 30 minutes for Monte Carlo).

**Process S — Homeostatic Sleep Pressure**

The homeostatic drive to sleep, normalised to [0, S_max]:

```
During wakefulness:  dS/dt = (S_max - S) / τ_wake
During sleep:        dS/dt = -S / τ_sleep
```

| Parameter | Value | Source |
|-----------|-------|--------|
| `S_max` | 1.0 | Normalised |
| `tau_wake` | 18.7 h | Daan et al. (1984) |
| `tau_sleep` | 4.2 h | ISS actigraphy, Barger et al. (2014) |

**Process C — Circadian Oscillator**

```
C(t) = M_c + A_c · cos(2π(t + φ₀ + Δφ) / T_c)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| `T_c` | 24.0 h | Circadian period |
| `M_c` | 0.50 | Oscillator mean |
| `A_c` | 0.17 | Amplitude |
| `phase_noise_amplitude` | 0.03 × A_c per orbit | Flynn-Evans et al. (2016) |

The ISS creates 16 light-dark cycles per day. This is modelled as a Wiener process on the circadian phase — independent Gaussian noise at each timestep with a slow-leak decay term (`× 0.97`) to prevent unbounded drift.

**Sleep Gates and Sleep Quality**

Sleep onset occurs when S rises above the upper circadian gate and the astronaut is inside the nominal sleep window (22:00–06:00 mission time). Sleep quality is derived geometrically from how well S sits between the upper and lower gates — fully centred produces quality 1.0; near either gate boundary degrades toward 0.1.

---

### Vestibular Mismatch Model

Implemented as `VestibularMismatchModel`. Based on Oman (1982) and Dai et al. (2011).

**Internal Model and Mismatch**

The brain maintains an internal prediction `ê(t)` of the expected otolith signal. The mismatch drives both adaptation and sickness:

```
m(t) = s(t) - ê(t)
dê/dt = k_adapt(S_norm) · m(t)
```

At microgravity entry, `s(t)` immediately becomes 1.0 while `ê(t)` remains 0.0. The mismatch starts at maximum and decays as the brain adapts.

**The Novel Coupling**

The adaptation rate is gated by homeostatic sleep pressure:

```
k_adapt(S_norm) = k₀ · (1 - w_s · S_norm)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| `k_adapt_0` (k₀) | 0.18 h⁻¹ | Calibrated to 67% first-72h incidence (Heer & Paloski 2006) |
| `w_s` | 0.60 (Uniform[0.55, 0.75] in Monte Carlo) | Model parameter — see limitations |

At maximum sleep deprivation (S_norm = 1.0), the effective adaptation rate drops to `k₀ × (1 - 0.60) = 0.072 h⁻¹` — less than half the rested baseline. Adaptation that would take ~6 hours rested takes ~14 hours sleep-deprived.

**Motion Sickness Onset Probability**

Onset probability per timestep follows a Michaelis-Menten saturation curve applied to the cumulative mismatch integral:

```
P(onset per hour) = σ · ∫|m|dt / (ξ + ∫|m|dt)
```

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `sigma_ms` (σ) | 0.40 | Maximum asymptotic onset rate |
| `xi_ms` (ξ) | 0.25 | Half-saturation constant |

**Vestibulo-Cardiac Reflex**

HR contribution from mismatch: `hr_delta = hr_gain × |m(t)|`, applied at 10% weighting to the state HR each step, with `hr_gain = 8.0 bpm per unit mismatch`.

---

### Fatigue Accumulation Model

Implemented as `FatigueModel`. The fatigue index F(t) is bounded on [0, 10], corresponding to the Samn-Perelli scale used in aerospace medicine.

**The ODE**

```
dF/dt = α · sleep_debt^1.2  +  β · |m(t)|^1.5  -  γ(C) · sleep_quality  +  ε
```

| Term | Parameter | Value | Meaning |
|------|-----------|-------|---------|
| Sleep debt accumulation | `alpha_sleep_debt` (α) | 0.35 h⁻¹ | Rate at which sleep debt drives fatigue |
| Vestibular stress | `beta_mismatch` (β) | 0.25 h⁻¹ | Rate at which mismatch drives fatigue |
| Circadian-gated recovery | `gamma_recovery_base` (γ) | 0.20 h⁻¹ | Base recovery rate during sleep |
| Circadian boost | `gamma_circadian_boost` | 0.10 h⁻¹ | Extra recovery at circadian nadir |
| Noise | ε ~ Gamma(0.5, 0.03) | — | Biological variability |

The superlinear exponents (1.2 on sleep debt, 1.5 on mismatch) encode the dose-response non-linearity observed in sleep deprivation studies. Recovery is circadian-gated — it is most efficient when the circadian oscillator C(t) is at its trough.

External event forcing (from active `ExerciseStressEvent`s) is injected into the ODE via the `fatigue_forcing` parameter at the `PhysicsEngine` level so the ODE state variable F remains authoritative across timesteps.

**Stress Formula**

The scalar stress state at each timestep is computed as:

```
stress(t) = 0.12 + circadian_stress(t) + min(0.45, F/10 × 0.60) + Σ event_stress_delta
```

where `circadian_stress` follows a sinusoidal profile and `event_stress_delta` aggregates contributions from all concurrently active events (motion sickness and exercise stress). This replaces the earlier EVA-only stress collection, ensuring that `MotionSicknessEvent` stress contributions are not silently overwritten.

---

### Coupling Engine

Implemented in `core/coupling_engine.py` as `CouplingEngine` (per-step) and `CouplingDiagnostics` (post-hoc counterfactual).

`CouplingEngine.update()` is called once per timestep immediately after `PhysicsEngine.step()`. It accumulates:

- `k_suppress` — fractional suppression of adaptation capacity at this step
- joint high-risk window tracking (fatigue AND high SMS probability simultaneously)
- per-step coupling probability traces (coupled, independent estimate, excess)

**Counterfactual Analysis**

The counterfactual question: *if vestibular adaptation had always run at the rested baseline rate k₀ (i.e., if sleep deprivation had no effect on adaptation), how much lower would the motion sickness risk have been?*

`CouplingDiagnostics.analyse()` computes both the actual (coupled) cumulative mismatch trajectory and an estimated counterfactual trajectory — scaled by the ratio `k_actual / k₀` at each step — then applies the Michaelis-Menten formula to both to get `P_coupled(t)` and `P_independent(t)`. The difference is the excess risk attributable to the coupling.

| Metric | Description |
|--------|-------------|
| `mean_excess_p_ms` | Average additional motion sickness probability per step from coupling |
| `peak_excess_p_ms` | Maximum point-in-time excess risk |
| `relative_excess_pct` | Excess as percentage of independent baseline |
| `joint_risk_excess_fraction` | Additional fraction of mission time in joint high-risk state |
| `mean_k_suppress` | Average fractional suppression of adaptation rate across mission |
| `time_high_coupling_frac` | Fraction of mission where coupling suppresses >40% of adaptation capacity |

Coupling diagnostics are attached to the risk report under `risk_report.coupling_diagnostics` and are available via the `/api/data/{run_id}/risk_report` endpoint.

> **Note**: The counterfactual currently uses a post-hoc scalar approximation rather than running a second parallel independent simulation. This systematically underestimates the independent baseline. See [Known Limitations](#known-limitations).

---

## Monte Carlo Engine

Implemented in `analytics/monte_carlo.py`.

A single simulation produces one possible trajectory for one hypothetical astronaut. Monte Carlo simulation runs N independent trajectories with biological parameters sampled from inter-individual variability distributions, producing a distribution of possible mission outcomes rather than a single point estimate. The Monte Carlo endpoint operates at 30-minute timesteps for computational efficiency; N=50 completes in approximately 1–3 seconds.

### Inter-Individual Variability Distributions

| Parameter | Distribution | Range | Biological Meaning |
|-----------|-------------|-------|-------------------|
| `tau_wake` | Normal(18.2, 1.5) h | [12, 24] | How quickly homeostatic pressure builds during wakefulness |
| `tau_sleep` | Normal(4.2, 0.4) h | [2.5, 6.5] | How quickly pressure dissipates during sleep |
| `k_adapt_0` | Normal(0.18, 0.03) h⁻¹ | [0.08, 0.35] | Rested vestibular adaptation rate |
| `w_s` | Uniform(0.55, 0.75) | — | Sleep-pressure suppression weight on adaptation |
| `alpha` | Nominal × Uniform(0.75, 1.35) | — | Individual sensitivity to sleep debt |
| `gamma` | Nominal × Uniform(0.80, 1.25) | — | Individual sleep recovery efficiency |

`tau_wake`, `tau_sleep`, and `k_adapt_0` use Normal distributions because they are measurable physiological quantities with known population means and symmetric individual variation. `w_s` uses Uniform because it is not empirically measured — the bounds reflect biological reasoning, but no single paper provides a central estimate. `alpha` and `gamma` are scaled multiplicatively to preserve proportional variation regardless of the nominal baseline.

The same per-astronaut variability distributions are used in the main simulation when `num_astronauts > 1`, with each crew member seeded by a deterministic RNG derived from their index.

### Outputs

```
run_monte_carlo() → {
    risk_summary:     aggregate threshold metrics across all N runs
    coupling_summary: excess risk distribution (the paper's primary result)
    distributions:    per-metric lists for histogram plotting
    envelopes:        mean ± std time-series for fan charts (downsampled to 200 points)
    conclusions:      plain-English statistical summary strings
}
```

Envelope time-series are produced for: fatigue, sleep quality, S (homeostatic pressure), C (circadian), vestibular mismatch, k_adapt, and the three coupling probability traces (coupled, independent, excess).

---

## BioGears Integration

BioGears is not part of the main simulation loop. The ODE system runs continuously and generates the full physiological trajectory. BioGears is invoked **only on discrete stochastic events** to produce high-fidelity cardiovascular snapshots for that event window, and on a **chronic-stress trigger** when fatigue first crosses 6.0 on the Samn-Perelli scale.

### Invocation Flow

```
Event.get_biogears_perturbation()
  → BioGearsAdapter._build_stressor()     # apply fatigue amplification
  → BioGearsScenarioRunner.run()          # write XML → call bg-cli subprocess
  → BioGearsOutputParser.parse()          # read results CSV → numpy arrays
  → BioGearsAdapter._scale_to_twin_state() # normalise to twin units
  → AstronautState.update(t, **response)  # inject into trajectory
```

The async wrapper runs BioGears in a thread pool executor to avoid blocking FastAPI's event loop.

### Action Mapping

| Event Type | BioGears Action | Key Parameters |
|-----------|----------------|----------------|
| Motion Sickness | `AcuteStressData` | `Severity` = nausea_severity × fatigue_amplification |
| EVA / Exercise | `ExerciseData > GenericExercise > Intensity` | `Intensity` = exercise_intensity (not amplified at input) |
| Sleep Disruption | `SleepData On/Off` + `PatientAssessmentRequestData PsychomotorVigilanceTask` | Duration extended by fatigue |
| Chronic Stress | `ExerciseData > GenericExercise > Intensity` | Fixed intensity = 0.3, triggered on first fatigue ≥ 6.0 crossing |

### Fatigue Amplification

Before building a stressor, the adapter scales parameters based on current fatigue level:

```
fatigue_norm  = fatigue_index / 10.0
amplification = 1.0 + 0.4 × fatigue_norm   # up to 1.40× at max fatigue
```

For **motion sickness**: severity is multiplied by amplification; duration extended by `1 + 0.2 × fatigue_norm`.  
For **exercise**: input intensity is unchanged (preserves BioGears scenario integrity); output HR is scaled by `1 + 0.25 × fatigue_norm` after parsing.  
For **sleep disruption**: no severity change; duration extended by `1 + 0.15 × fatigue_norm`.

### Scenario Structure

Each scenario applies a 30-second stabilisation advance before the stressor. Motion sickness scenarios include a 60-second recovery observation window after stressor removal. Exercise scenarios apply a 90-second post-exercise recovery window. Sleep disruption scenarios include a 60-second post-wake stabilisation and a PVT assessment request. All scenarios cap simulated duration at 10 minutes regardless of the event duration drawn from the stochastic model.

### Signals Collected

All scenarios request 8 signals at 0.05 samples/second (1 sample per 20 seconds):

`HeartRate` · `MeanArterialPressure` · `SystolicArterialPressure` · `DiastolicArterialPressure` · `OxygenSaturation` · `RespirationRate` · `TidalVolume` · `CoreTemperature`

### Patient Model

Every scenario uses `StandardMale.xml` loaded from the pre-computed equilibrium state `states/StandardMale@0s.xml`. Inter-individual variability is handled at the ODE layer through Monte Carlo parameter sampling, not at the BioGears layer.

### Mock Mode

When `bg-cli.exe` is not present, the runner automatically generates synthetic CSVs with physiologically distinct profiles:

- **Motion sickness**: Gaussian spike peaking at t/3 of scenario duration
- **Exercise**: Trapezoid ramp — 20% ramp-up, plateau, 20% ramp-down
- **Sleep disruption**: Flat suppressed baseline (HR 58 bpm, MAP 80 mmHg, RR 12 bpm)

---

## API Reference

The backend is a FastAPI application served by Uvicorn.

### Base URL

```
http://localhost:8000
```

Interactive documentation is available at `/docs` (Swagger UI) and `/redoc`.

### Router Groups

| Prefix | Description |
|--------|-------------|
| `/api/simulation` | Simulation lifecycle — start, poll, stop, delete, list, results, Monte Carlo, sensitivity |
| `/api/data` | Retrieve completed outputs — full trajectories, risk reports, trend analyses |
| `/api/health` | Health checks, dependency verification, Kubernetes liveness/readiness probes |
| `/api/config` | Serves Groq API key from server environment to frontend |

### Key Endpoints

```
POST   /api/simulation/run                        Start a new simulation, returns run_id immediately
GET    /api/simulation/status/{run_id}            Poll simulation progress
GET    /api/simulation/list                       Paginated list of all runs (filterable by status)
POST   /api/simulation/stop/{run_id}              Stop a running simulation
DELETE /api/simulation/delete/{run_id}            Delete a simulation run and its data
GET    /api/simulation/config/{run_id}            Retrieve the config used for a run
POST   /api/simulation/monte-carlo               Run Monte Carlo batch (synchronous, ~1-3 s for N=50)
GET    /api/simulation/sensitivity/{variable}     Sensitivity profiles for heart_rate or sleep_quality
GET    /api/data/{run_id}/results                 Full trajectory data
GET    /api/data/{run_id}/risk_report             Threshold analysis, at-risk windows, coupling diagnostics
GET    /api/data/{run_id}/trend_analysis          Linear trend detection on state variables
POST   /api/simulation/ai/chat                    Groq AI proxy (CORS bypass, Llama 3.3 70B)
GET    /api/health/                               Basic health check
GET    /api/health/ping                           Ultra-lightweight ping for load balancers
GET    /api/health/status                         Detailed system status with resource usage
GET    /api/health/liveness                       Kubernetes liveness probe
GET    /api/health/readiness                      Kubernetes readiness probe
GET    /api/health/metrics                        Prometheus-style process metrics
GET    /api/health/dependencies                   Dependency version inventory
```

### SimulationConfig Schema

Key fields accepted by `POST /api/simulation/run`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mission_duration_hours` | float | 720.0 | Simulation duration (1 h – 8760 h) |
| `time_step_minutes` | float | 5.0 | ODE integration timestep (0.1 – 60 min) |
| `astronaut_id` | string | "default" | Identifier for baseline profiles |
| `baseline_hr` | float | 75.0 | Baseline heart rate (bpm) |
| `baseline_sleep_quality` | float | 0.8 | Baseline sleep quality [0–1] |
| `initial_fatigue` | float | 0.0 | Initial fatigue index [0–10] |
| `num_astronauts` | int | 1 | Crew size to simulate (1–5) |
| `enable_motion_sickness` | bool | true | Enable stochastic SMS events |
| `enable_sleep_disruption` | bool | true | Enable sleep disruption events |
| `enable_exercise_stress` | bool | true | Enable EVA/exercise stress events |
| `use_biogears` | bool | true | Enable BioGears cardiovascular responses |
| `save_trajectories` | bool | true | Persist full state time-series |
| `save_events` | bool | true | Persist event logs |

### Middleware

- **CORS**: `allow_origins=["*"]` — open for development
- **GZip**: Compresses responses larger than 1 KB
- **Request timing**: `X-Process-Time` header on every response
- **Request tracing**: `X-Request-ID` header on every response

---

## Frontend Dashboard

A Jinja2 template-based web application with modular JavaScript.

### Technology Stack

| Component | Technology |
|-----------|-----------|
| 3D visualisation | Three.js r160 (GLTF models: `Astronaut.glb`, `Endurance.glb`, `BlackHole.glb`) |
| Charts | Chart.js 4.4 |
| AI risk explanation | Groq API (Llama 3.3 70B, proxied through `/api/simulation/ai/chat`) |
| PDF export | jsPDF (fully client-side, no server involvement) |
| Audio | Hans Zimmer — *No Time For Caution* (`static/No Time For Caution.mp3`) |

### JavaScript Modules

| File | Responsibility |
|------|---------------|
| `api.js` | HTTP client for all backend communication |
| `simulation.js` | Simulation lifecycle state machine |
| `playback.js` | Post-simulation trajectory playback engine with pause/resume |
| `charts.js` | Real-time and scrubbing chart management |
| `montecarlo.js` | Monte Carlo results display and envelope charts |
| `ai-chat.js` | Groq AI integration for natural-language risk explanation |
| `export.js` | PDF report generation |
| `3d-viewer.js` | Three.js scene, model loading, physiological state → scene mapping, click-to-inspect raycaster |
| `tour.js` | 6-step onboarding tour |
| `debug-panel.js` | Developer positioning tool (Ctrl+Shift+D) |
| `utils.js` | Shared utilities and constants |

### 3D Viewer Features

The Three.js scene renders the Endurance spacecraft with an interior habitat containing the astronaut model. Camera modes toggle between ship-exterior orbit (`ship`) and astronaut-interior view (`astro`). Risk state is reflected in scene lighting — red ambient shift and tremor effects activate at HIGH/CRITICAL fatigue. A click-to-inspect raycaster shows physiological readings for the head (stress), upper torso (heart rate, SpO₂), and lower body (fatigue, motion severity) regions of the astronaut mesh. Risk-level transitions trigger audio tones via the Web Audio API.

### Template Partials

Twelve HTML partials compose the dashboard layout:

`api-bar` · `charts` · `controls` · `full-chart-modal` · `header` · `load-modal` · `mission-risk` · `montecarlo` · `run-comparison` · `status-panel` · `timeline` · `twin-panel`

The `run-comparison` partial enables side-by-side display of multiple completed simulation runs for direct trajectory and risk metric comparison.

### PDF Report Contents

Generated entirely in-browser; includes: cover page, statistical summary badges, full time-series charts as embedded PNG images, fatigue analysis table, sleep quality analysis table, risk windows section, vestibular-fatigue coupling diagnostics, AI-generated or rule-based conclusions, and a flight surgeon signature block. Filename includes run ID and generation date.

---

## Installation

### Prerequisites

- Python 3.10 or later
- Node.js not required (all frontend dependencies are CDN-loaded)
- BioGears 8.x (optional — system runs in mock mode without it)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/astronaut-digital-twin.git
cd astronaut-digital-twin

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys (see Configuration section)

# Run the server
python main.py
```

The server starts at `http://localhost:8000`. The dashboard is served at the root URL.

### BioGears Setup (Optional)

If BioGears is installed, point the adapter to the `bin/` directory:

```bash
BG_CLI_PATH=C:\path\to\biogears\bin   # Windows
BG_CLI_PATH=/opt/biogears/bin         # Linux
```

The XSD schema validation file at `bin/xsd/BioGearsDataModel.xsd` should be renamed to disable it — validation causes namespace errors with the generated scenario XML:

```bash
# Windows
rename "C:\biogears\bin\xsd\BioGearsDataModel.xsd" BioGearsDataModel.xsd.bak

# Linux
mv /opt/biogears/bin/xsd/BioGearsDataModel.xsd /opt/biogears/bin/xsd/BioGearsDataModel.xsd.bak
```

### CLI Options

```
python main.py [OPTIONS]

Options:
  --host       Host to bind to (default: 0.0.0.0)
  --port       Port to bind to (default: 8000)
  --reload     Enable auto-reload for development
  --workers    Number of worker processes (default: 1)
  --log-level  Logging level: debug|info|warning|error|critical
```

---

## Configuration

### Environment Variables (`.env`)

```env
GROQ_API_KEY=your_groq_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
BG_CLI_PATH=C:\Users\username\biogears\bin
```

`GROQ_API_KEY` is served to the frontend at page load via `/api/config` and is used directly by the Groq chat proxy. `ANTHROPIC_API_KEY` is available for server-side Claude integrations. Neither key is embedded in the HTML source.

### `config/simulation_config.yaml`

Default mission parameters and model parameter overrides. Loaded at startup and accessible through the simulation manager.

### `config/distributions.yaml`

Statistical distributions for all probabilistic variables, each with explicit academic citations. Includes:

| Variable | Distribution | Parameters |
|----------|-------------|-----------|
| Heart rate | Normal | μ=75 bpm, σ=5 bpm |
| Sleep quality | Beta | α=5, β=2 (mean ≈ 0.71) |
| Motion sickness onset | Poisson | λ=0.03 events/h (time-inhomogeneous, α=0.1/h decay) |
| Motion sickness severity | Beta | α=2, β=3 (mean = 0.4) |
| Motion sickness duration | Gamma | shape=2.5, scale=0.8 h |
| Fatigue noise | Gamma | shape=2, scale=0.1 |
| Stress response | LogNormal | μ=0.5, σ=0.3 |
| Recovery time | Weibull | shape=1.5, scale=2.0 h |

---

## File Structure

```
.
├── __init__.py                   # Version: v1.1.0
│
├── core/
│   ├── fatigue_model.py          # BorbelyModel, VestibularMismatchModel, FatigueModel, PhysicsEngine
│   ├── coupling_engine.py        # CouplingEngine (per-step diagnostics) + CouplingDiagnostics (counterfactual)
│   ├── probabilistic_models.py   # ProbabilisticModels — sampling from distributions.yaml
│   └── state_manager.py          # AstronautState — all time-series arrays for one simulation run
│
├── analytics/
│   ├── monte_carlo.py            # Monte Carlo engine with inter-individual variability
│   ├── risk_engine.py            # Threshold analysis, at-risk window detection, cumulative load
│   └── trend_analysis.py         # Linear trend detection on physiological variables
│
├── events/
│   ├── base_event.py             # Abstract Event base class (EventPriority, EventStatus, EventEffect)
│   ├── event_scheduler.py        # Priority queue event manager with disabled-event filtering
│   ├── motion_sickness_event.py  # Stochastic SMS events from ODE p_ms_step probability
│   ├── sleep_disruption_event.py # Sleep window disruption events
│   └── exercise_stress_event.py  # EVA/exercise stress events (ExerciseData → BioGears)
│
├── biogears/
│   ├── __init__.py               # Package exports: BioGearsAdapter, ScenarioRunner, OutputParser
│   ├── biogears_adapter.py       # High-level async bridge: digital twin ↔ BioGears
│   ├── scenario_runner.py        # XML scenario builder + bg-cli subprocess wrapper + mock mode
│   └── output_parser.py          # BioGears CSV parser, column normalisation, time-axis alignment
│
├── api/
│   ├── dependencies.py           # SimulationManager, ConfigLoader, FastAPI dependency injection
│   └── routes/
│       ├── simulation.py         # Lifecycle + Monte Carlo + sensitivity + Groq AI proxy
│       ├── data.py               # Results and analytics retrieval
│       ├── health.py             # Health checks, Kubernetes probes, Prometheus metrics
│       └── config.py             # API key serving endpoint
│
├── templates/
│   ├── base.html                 # Base layout with script loading
│   ├── index.html                # Page assembly (includes all partials)
│   └── partials/                 # 12 HTML partial components
│       ├── api-bar.html
│       ├── charts.html
│       ├── controls.html
│       ├── full-chart-modal.html
│       ├── header.html
│       ├── load-modal.html
│       ├── mission-risk.html
│       ├── montecarlo.html
│       ├── run-comparison.html
│       ├── status-panel.html
│       ├── timeline.html
│       └── twin-panel.html
│
├── static/
│   ├── js/                       # 11 JavaScript modules
│   ├── style.css                 # Application stylesheet
│   ├── models/                   # GLTF 3D models (Astronaut, Endurance, BlackHole)
│   └── No Time For Caution.mp3   # Background music
│
├── config/
│   ├── simulation_config.yaml    # Mission and model parameters
│   └── distributions.yaml        # Probabilistic distribution definitions with citations
│
├── utils/
│   ├── helpers.py                # Version, git revision, utility functions
│   └── logger.py                 # Structured logging setup
│
└── main.py                       # FastAPI application entry point, middleware, router registration
```

---

## Academic References

### Sleep Science

- Borbely, A.A. (1982). A two process model of sleep regulation. *Human Neurobiology*, 1(3), 195–204.
- Daan, S., Beersma, D.G.M., and Borbely, A.A. (1984). Timing of human sleep: Recovery process gated by a circadian pacemaker. *American Journal of Physiology*, 246(2), R161–R183.
- Barger, L.K., et al. (2014). Prevalence of sleep deficiency and use of hypnotic drugs in astronauts before, during, and after spaceflight. *The Lancet Neurology*, 13(9), 904–912. — Primary source for ISS sleep efficiency (mean 0.71) and actigraphy parameters.
- Van Dongen, H.P.A., et al. (2003). The cumulative cost of additional wakefulness. *Sleep*, 26(2), 117–126. — Source for cumulative fatigue dose-response calibration.
- Flynn-Evans, E.E., et al. (2016). Circadian misalignment affects sleep and medication use before and during spaceflight. *npj Microgravity*, 2, 15013. — Source for ISS circadian phase drift standard deviation (0.08 h/day).

### Vestibular Physiology and Space Motion Sickness

- Oman, C.M. (1982). A heuristic mathematical model for the dynamics of sensory conflict and motion sickness. *Acta Otolaryngologica*, Supplement 392, 44. — Foundational paper for the vestibular mismatch model.
- Dai, M., et al. (2011). The relation of motion sickness to the spatial-temporal properties of velocity storage. *Experimental Brain Research*, 210(1), 45–64.
- Heer, M. and Paloski, W.H. (2006). Space motion sickness: Incidence, etiology, and countermeasures. *Autonomic Neuroscience*, 129(1–2), 77–79. — Source for the 67% first-72h SMS incidence rate used to calibrate σ and ξ.
- Reschke, M.F., et al. (2018). Space motion sickness: A synthesis of recent studies. *Journal of Vestibular Research*, 28(2), 99–109.

### Standards and Technical References

- NASA-STD-3001. NASA Space Flight Human-System Standard. — Basis for risk threshold values (fatigue > 7.0 = HIGH risk, sleep quality < 0.40 = critical).
- Hamilton, D.R., Murray, J.D., and Ball, C.G. (2011). Cardiac health for astronauts. *Canadian Journal of Cardiology*, 27(3), e24–e31. — Source for microgravity baseline HR (75 bpm).

---

## Known Limitations

### Counterfactual Approximation

The most significant scientific limitation. The current counterfactual analysis uses a post-hoc scalar approximation — scaling the actual cumulative mismatch by the `k_actual/k₀` ratio — rather than running an independent parallel simulation with k_adapt held constant at k₀. Because the mismatch dynamics are nonlinear and the approximation assumes linearity, this systematically underestimates the independent baseline, making the excess risk appear larger than the correct value. A proper implementation for journal submission requires a second parallel trajectory per Monte Carlo run, doubling computational cost but producing the scientifically defensible result.

### Uncited Nonlinear Exponents

The fatigue ODE uses superlinear exponents of 1.2 (sleep debt term) and 1.5 (vestibular mismatch term). These values are physiologically plausible and consistent with observed dose-response non-linearity, but are not directly cited to a specific paper. A sensitivity analysis demonstrating that key outputs are robust to variation in these exponents within a physiologically reasonable range is required before submission.

### The `w_s` Parameter

The sleep-pressure coupling weight `w_s = 0.60` is the most important and least constrained free parameter in the model. There is no single paper providing a direct empirical measurement of how much vestibular adaptation capacity is lost per unit of homeostatic sleep pressure. A sensitivity analysis sweeping `w_s` from 0.4 to 0.9 is essential to determine whether the coupling excess risk result is robust to this uncertainty or critically dependent on it.

### Validation Against Real Data

The model currently reproduces known summary statistics from the literature (67% first-72h SMS incidence, 0.71 mean sleep efficiency, appropriate Samn-Perelli fatigue levels) but has not been formally validated against individual-level ISS crew physiological data. Comparison against actigraphy data from NASA's Life Sciences Data Archive would significantly strengthen the research claims. Individual-level ISS data is publicly available to researchers upon application.

### Patient Model

BioGears runs all scenarios against a single `StandardMale.xml` patient. A female patient variant or a patient parameterised from individual astronaut anthropometrics and physiology would improve the physiological realism of BioGears-generated cardiovascular responses. The current approach relies entirely on the ODE layer for inter-individual variability.

### Stress Formula Double-Counting Risk

The stress computation aggregates contributions from all concurrently active events using a general collector (`_collect_event_stress_deltas`). An EVA safety net is included to avoid double-counting exercise stress if it was already captured via the effects list. However, in edge cases where multiple event types fire in the same timestep, the interaction between the safety net heuristic and the effects list may produce minor inaccuracies. A cleaner resolution would be to have each event type own its exclusive stress channel.

---