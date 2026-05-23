// Shared terrain math. `sampleTerrain` is the canonical seabed height
// function — both Ground (mesh build) and Seagrass (placement) call it,
// so they agree on where the seabed is.

// Noise sample frequency in world units. Lower = broader dunes; higher =
// busier ripples.
export const NOISE_FREQUENCY = 0.08;

// Fixed seed so the heightfield is identical across reloads — lets other
// scene elements be authored against a known seabed.
export const SEED = 1;

// Three named height strata blended in fBM-space. Slope between strata
// at a given point is (H_next - H_prev)/(T_hi - T_lo) * |grad fbm|, so
// terrain reads gradual where the noise gradient is low and steeper
// where it spikes — without ever forming a hard cliff.
export const H_BASIN = 0.2;
export const H_SHELF = 3;
export const H_REEF = 5.5;

// Stratum thresholds in fBM-space ([0, 1]). LO/HI bracket the
// transition; outside the bracket the stratum value is constant.
export const T_BASIN_SHELF_LO = 0.3;
export const T_BASIN_SHELF_HI = 0.48;
export const T_SHELF_REEF_LO = 0.55;
export const T_SHELF_REEF_HI = 0.75;

// Domain warp.
export const WARP_STRENGTH = 1.4;
export const WARP_FREQUENCY = 0.5;

// Slow tilt + fine grain added after stratum blending.
export const TILT_FREQUENCY = 0.15;
export const TILT_AMPLITUDE = 1.0;
export const DETAIL_FREQUENCY = 2.5;
export const DETAIL_AMPLITUDE = 0.1;

// Soft asymptotic ceiling for above-water mounds. Absolute Y in the
// Ground mesh's local frame — tune so it sits just above WATER_Y in
// Scene; the softness controls how quickly tall noise spikes asymptote
// into the flat island top.
export const ISLAND_TOP_Y = 5.8;
export const ISLAND_PLATEAU_SOFTNESS = 0.12;

function hash2D(x: number, y: number, seed: number): number {
    const s = Math.sin(x * 127.1 + y * 311.7 + seed * 0.013) * 43758.5453;
    return s - Math.floor(s);
}

function fade(t: number): number {
    return t * t * (3 - 2 * t);
}

function valueNoise(x: number, y: number, seed: number): number {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const fx = x - ix;
    const fy = y - iy;
    const a = hash2D(ix, iy, seed);
    const b = hash2D(ix + 1, iy, seed);
    const c = hash2D(ix, iy + 1, seed);
    const d = hash2D(ix + 1, iy + 1, seed);
    const ux = fade(fx);
    const uy = fade(fy);
    return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy;
}

// Four-octave fractional Brownian motion. Each octave doubles the
// frequency and halves the amplitude, with a per-octave seed offset so
// the layers don't align into visible grid artifacts.
export function fbm(x: number, y: number, seed: number): number {
    let v = 0;
    let amp = 0.5;
    let freq = 1;
    for (let i = 0; i < 4; i++) {
        v += amp * valueNoise(x * freq, y * freq, seed + i * 17);
        amp *= 0.5;
        freq *= 2;
    }
    return v;
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
    const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}

export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

// One-sided smooth cap: h - softness · softplus((h - cap) / softness).
// Below `cap` returns ≈ h; far above asymptotes to `cap`.
export function softCap(h: number, cap: number, softness: number): number {
    const t = (h - cap) / softness;
    const softplus = t > 20 ? t : Math.log1p(Math.exp(t));
    return h - softness * softplus;
}

// Returns the seabed height at world (x, z) in the Ground mesh's local
// frame, plus the stratum mix value n ∈ [0, 1]. `stratum` is the raw
// (warped) fBM value used to choose between basin / shelf / reef —
// foliage placement can reject candidates by stratum range without
// having to invert the height function.
export function sampleTerrain(
    x: number,
    z: number,
): { height: number; stratum: number } {
    const warpX =
        (fbm(x * WARP_FREQUENCY, z * WARP_FREQUENCY, SEED + 11) - 0.5) *
        2 *
        WARP_STRENGTH;
    const warpZ =
        (fbm(x * WARP_FREQUENCY, z * WARP_FREQUENCY, SEED + 23) - 0.5) *
        2 *
        WARP_STRENGTH;
    const wx = x + warpX;
    const wz = z + warpZ;
    const n = fbm(wx * NOISE_FREQUENCY, wz * NOISE_FREQUENCY, SEED);
    let h = H_BASIN;
    h = lerp(h, H_SHELF, smoothstep(T_BASIN_SHELF_LO, T_BASIN_SHELF_HI, n));
    h = lerp(h, H_REEF, smoothstep(T_SHELF_REEF_LO, T_SHELF_REEF_HI, n));
    h +=
        (fbm(x * TILT_FREQUENCY, z * TILT_FREQUENCY, SEED + 47) - 0.5) *
        2 *
        TILT_AMPLITUDE;
    h +=
        (fbm(x * DETAIL_FREQUENCY, z * DETAIL_FREQUENCY, SEED + 71) - 0.5) *
        2 *
        DETAIL_AMPLITUDE;
    h = softCap(h, ISLAND_TOP_Y, ISLAND_PLATEAU_SOFTNESS);
    return { height: h, stratum: n };
}
