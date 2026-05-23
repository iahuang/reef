// Procedural ocean-current vector field. Returns a 3D velocity at any
// (x, y, z, t) in noise-world coords — divergence-free in the XZ plane
// (curl of a 2D scalar potential) so streamlines naturally circle into
// slow eddies rather than draining toward sources/sinks. A constant
// drift bias is added so the average flow has a definite direction
// across the map. Y carries a weak independent component so things in
// the current don't sit at a perfectly fixed depth.

import { fbm, SEED } from "./terrain";

// Seed offsets keep the current field uncorrelated with terrain noise.
const SEED_POTENTIAL = SEED + 4001;
const SEED_VERTICAL = SEED + 4017;

// Spatial frequency of the streamfunction. Lower = larger, smoother
// eddies that span many world units. The map is 60 wide (2 × half
// extent of 30), so a frequency near 0.04 gives 1–2 large gyres across
// the world at a time.
const POTENTIAL_FREQUENCY = 0.04;
// Slow temporal drift of the potential — the field morphs over minutes
// rather than seconds so currents have a sense of persistence.
const POTENTIAL_TIME_FREQUENCY = 0.025;
// Strength of the curl-derived horizontal flow (world units / second).
const CURL_STRENGTH = 0.9;
// Finite-difference step for the curl derivatives. Large enough that
// successive samples land in different valleys of the fbm, small
// relative to the eddy scale (1 / POTENTIAL_FREQUENCY ≈ 25).
const CURL_EPS = 0.6;

// Constant background drift. Gives the field a preferred direction so
// drifters traverse the map instead of just orbiting in place.
const DRIFT_X = 0.18;
const DRIFT_Z = 0.12;

// Weak vertical component. Independent from the horizontal field so up-
// and down-wellings drift through the column without being tied to the
// horizontal eddy structure.
const VERTICAL_FREQUENCY = 0.07;
const VERTICAL_TIME_FREQUENCY = 0.04;
const VERTICAL_STRENGTH = 0.18;

// Sample the streamfunction ψ(x, z, t). Curl in 2D is
// (∂ψ/∂z, -∂ψ/∂x), which is automatically divergence-free — good for
// flow visualisation because streamlines close and the field doesn't
// secretly pump mass toward sinks.
function potential(x: number, z: number, t: number): number {
    return fbm(
        x * POTENTIAL_FREQUENCY + t * POTENTIAL_TIME_FREQUENCY,
        z * POTENTIAL_FREQUENCY,
        SEED_POTENTIAL,
    );
}

// Fill `out` with the current vector at (x, y, z) and absolute time t.
// Caller-owned scratch buffer so the per-frame inner loop allocates
// nothing.
export function sampleCurrent(
    x: number,
    _y: number,
    z: number,
    t: number,
    out: [number, number, number],
): void {
    const pXp = potential(x + CURL_EPS, z, t);
    const pXm = potential(x - CURL_EPS, z, t);
    const pZp = potential(x, z + CURL_EPS, t);
    const pZm = potential(x, z - CURL_EPS, t);
    const inv2eps = 1 / (2 * CURL_EPS);
    const dPsi_dz = (pZp - pZm) * inv2eps;
    const dPsi_dx = (pXp - pXm) * inv2eps;
    // Curl of (0, ψ, 0): horizontal flow = (∂ψ/∂z, -∂ψ/∂x).
    // Multiply by 2 / POTENTIAL_FREQUENCY ≈ characteristic length so
    // CURL_STRENGTH reads as a speed in world units / second rather
    // than depending on the spatial frequency of the potential.
    const scale = CURL_STRENGTH * (2 / POTENTIAL_FREQUENCY);
    out[0] = dPsi_dz * scale + DRIFT_X;
    out[2] = -dPsi_dx * scale + DRIFT_Z;

    const vy = fbm(
        x * VERTICAL_FREQUENCY,
        z * VERTICAL_FREQUENCY + t * VERTICAL_TIME_FREQUENCY,
        SEED_VERTICAL,
    );
    out[1] = (vy - 0.5) * 2 * VERTICAL_STRENGTH;
}
