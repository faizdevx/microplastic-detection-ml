# Scattering model

What `src/mp/scatter/` computes, with its assumptions and limits. Every constant is in
`docs/literature_params.csv` (with a source) or is a rig parameter that has not been
measured (`UNMEASURED`); those parameters are set in `configs/scattering.yaml` and swept by
`mp scatter-analyse`.

## Geometry

A 650 nm laser beam crosses a square cuvette (default path L = 10 mm) filled with a
suspension. Four BPW34 photodiodes sit in the horizontal plane at angles
θ1..θ4 from the beam (default 20°, 45°, 90°, 135°), a distance D from the cuvette centre
(default 50 mm). Each sees the solid angle Ω = A / D² (A = 7.5 mm²). Detector acceptance is
handled by averaging over five polar offsets within the half-angle sqrt(Ω/π).

## Particles

Homogeneous spheres, relative index m = (n_p - i k_p) / n_water with k_p = 0, diameter d
from a number-weighted log-normal distribution (equal-probability quadrature nodes).
Size parameter x = π d n_water / λ0. Mie coefficients from `miepython`; the angular
amplitudes S1, S2 are evaluated with the standard Bohren and Huffman recursion in numpy, which
is checked against Q_sca (integral of the angular intensity), Q_ext (optical theorem) and
published efficiencies in `tests/test_scatter.py`.

Differential cross section (unpolarised): dσ/dΩ = (|S1|² + |S2|²) / (2 k²), k = 2π n_water / λ0.

## Concentration to number density

1 mg/L = 1 g/m³. N = C / (ρ · (π/6) · ⟨d³⟩), with ⟨d³⟩ averaged over the size nodes.

## Detector power

Single scattering with attenuation:

    P_i = P_laser · N · L · ⟨dσ/dΩ(θ_i)⟩ · Ω · exp(-μ L),   μ = N σ_ext   (`none`)

`first_order` replaces σ_ext in the exponent by the transport cross section σ_ext - g σ_sca.
`monte_carlo` follows photons through the cuvette (forced collisions, scattering angles
drawn from the size-averaged Mie phase function, next-event estimate for each detector,
transmission to the wall along the detector direction). At low optical depth it reproduces
the single-scattering result (tested); above τ ≈ 0.1 forward detectors gain signal from
repeated small-angle scattering and the forward/backward ratio changes.

## Detector chain

Photocurrent = responsivity · power + dark current; transimpedance gain R_f; noise terms
(shot 2qIB, Johnson 4kTB/R_f, common-mode laser drift, ambient offset, Poisson fluctuation of
the number of particles in the illuminated volume, with `independent_volumes` fresh
volumes per reading); ADC quantisation floor(V / V_fs · 2^bits) clipped to [0, 2^bits - 1].
Two ADC options: MCP3008 (10-bit, 3.3 V) and ADS1115 (15 usable bits at ±4.096 V).

## Assumptions and limits

- Spherical, homogeneous, non-absorbing particles; no aggregation, biofilm, pigments or fillers.
- Single polymer per sample; no non-plastic particles, bubbles or turbidity.
- Mass concentration to number density assumes the nominal polymer density.
- The multiple-scattering Monte Carlo uses an idealised point beam and a perfectly absorbing
  wall; the detector acceptance is a 1-D polar average, not a full solid-angle integral.
- Stray light, dark offset and ambient light are constants per geometry.
- Rig values (angles, distance, gain, laser power, ADC, path, beam size, number of
  independent volumes) are not measured. Results are meaningful as dependences on these
  parameters, not as absolute counts.
