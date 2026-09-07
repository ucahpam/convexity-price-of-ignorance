# Convexity and the Price of Ignorance

Code accompanying the MSc dissertation *Convexity and the Price of
Ignorance: Finite Element Valuation of European Options under an
Uncertain Volatility Risk Premium* (Paul Michel, UCL Department of
Mathematics).

The thesis prices European options under the Heston stochastic-volatility
model when the market price of volatility risk λ is known only to lie in
a compact interval, following the worst-case framework of Jaroszkowski &
Jensen (2023). The worst- and best-case prices are the solutions of a
nonlinear Hamilton–Jacobi–Bellman equation, solved with the monotone
finite element scheme of the FEISol library. This repository contains
the problem-specific layer written for the thesis: the contract
specifications, the experiment drivers, the parameter-study scripts, and
the post-processing that produces the figures.

## Requirements

The code runs against the legacy FEniCS / DOLFIN 2019.1.0 stack and the
FEISol library, inside the supervisor-provided Docker image. From the
FEISol root inside the container, a script is run with, for example:

```
docker exec -it -w /opt/FEISol feisol-demos python3 sweep_power.py
```

## Contents

### Contract specifications (the model)
| File | Contract |
|------|----------|
| `parameters_vanilla_call.py`   | European call |
| `parameters_vanilla_put.py`    | European put |
| `parameters_power_general.py`  | Powered call, variable degree `p` (used for the convexity sweep) |
| `parameters_power_call.py`     | Powered call, fixed `p = 2` |

Each is a `Parameters` class implementing the `ParametersBase` contract:
the coefficients of the transformed operator, the payoff, the control
set, and the boundary conditions. Pricing a new contract means writing a
new `Parameters` file; the solver is untouched.

### Drivers and parameter studies
| File | Role |
|------|------|
| `demo_vanilla.py`    | Run driver: selects the contract and the worst-/best-case problem |
| `sweep_power.py`     | Band vs payoff convexity (varies `p`) — the convexity sweep |
| `sweep_interval.py`  | Band vs width of the uncertainty interval L |
| `sweep_diameter.py`  | Value/Delta gaps vs control-set diameter |

### Verification and post-processing
| File | Role |
|------|------|
| `laplace_fenics.py`      | Manufactured-solution (Poisson) verification |
| `band_peak.py`           | Reads the peak band for the mesh-refinement study |
| `plot_control.py`        | Reconstructs the bang-bang optimal-control map |
| `plot_band.py`           | Uncertainty-band surfaces |
| `plot_delta.py`          | Hedging-Delta surfaces and slices |
| `plot_delta_surface.py`  | Delta-difference surfaces |
| `plot_nonlinearity.py`   | Nonlinearity (worst-case minus fixed-λ) |
| `plot_option.py`         | Value surfaces and slices |

## Parameters

Throughout, the Doran calibration is used: κ = 7, γ = 0.3, ξ = 0.7,
ρ = 0.5, r = 0.03, T = 0.5, with strike K = 1.60 on the shipped square
mesh and uncertainty interval L = [−2.4, −1.6].

## Note on the AI-assistance disclosure

In line with the project's guidelines, AI tools were used as a support
for debugging, drafting and iterating parameter files and plotting
scripts, and drafting document structure later rewritten by the author.
All runs were executed and checked by the author, and the experimental
designs, results and conclusions are the author's own. See the thesis
appendix for the full acknowledgement.
