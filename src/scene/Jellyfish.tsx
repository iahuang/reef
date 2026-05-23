import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { sampleTerrain } from "./terrain";
import { worldOffset, WORLD_HALF_EXTENT } from "./worldOffset";
import { sampleCurrent } from "./oceanCurrent";

// --- Motion tuning ---
// Jellies are passive drifters: the current dominates, separation
// nudges them apart, and walls / terrain steer them away. There is no
// alignment or cohesion — they don't follow each other.
const MAX_SPEED = 1.0;
const MIN_SPEED = 0.05;
const MAX_FORCE = 1.2;

// Loose social distance. PERCEPTION is the range at which a neighbour
// starts to register; SEPARATION is the closer band that actually pushes
// (jellies should drift past one another, not stick or repel from far).
const PERCEPTION = 1.4;
const SEPARATION = 0.9;

// Force weights. CURRENT dominates so the field actually drives motion.
// SEPARATION is light so close-passes nudge rather than swerve. BOUNDS
// and TERRAIN are firm — visual clipping into the world edge or seabed
// would be more wrong than a hard turn.
const W_CURRENT = 1;
const W_SEPARATION = 1.1;
const W_BOUNDS = 3.0;
const W_TERRAIN_UP = 6.0;
const W_TERRAIN_REDIRECT = 3.0;

// Soft margin band at the XZ walls of the navigable world. Larger than
// the fish margin — jellies are slower and a wider band gives them
// room to curve back without ever touching the limit.
const XZ_BOUND = WORLD_HALF_EXTENT - 0.5;
const XZ_MARGIN = 5.0;

// Vertical clearance over the seabed. Looser than fish (0.55) because
// the tentacles trail well below the bell origin — measured against
// the bell centre, we need extra room.
const TERRAIN_CLEARANCE = 1.3;
const TERRAIN_LOOKAHEAD = 0.9;
const TERRAIN_GRAD_EPS = 1.0;

// --- Visual / geometry ---
// Bell dome dimensions.
const BELL_RADIUS = 0.32 * 0.7;
const BELL_HEIGHT = 0.22 * 0.7;
const BELL_LAT_SEGMENTS = 8;
const BELL_LON_SEGMENTS = 16;

// Tentacles hanging from the bell rim. Few sides per tube — readable
// silhouette at this camera distance without paying for a smooth round.
const TENTACLE_COUNT = 8;
const TENTACLE_SEGMENTS = 4;
const TENTACLE_SIDES = 3;
const TENTACLE_LENGTH = 0.5;
const TENTACLE_BASE_RADIUS = 0.014;
const TENTACLE_TIP_RADIUS = 0.004;

// Bell pulse frequency (radians/sec). A full sin cycle is 2π, so 4.0
// gives ~0.64 Hz — slow, drifting, moon-jelly-like.
const PULSE_OMEGA = 4.0;
// Peak radial swell and vertical squash at the bell tip. The tip moves
// most, the rim barely.
const BELL_RADIAL_AMP = 0.16;
const BELL_VERT_AMP = 0.08;
// Tentacle undulation amplitude (world units) at the very tip. Tapers
// to zero at the attachment so the rim stays anchored to the bell.
const TENTACLE_SWAY = 0.06;

// --- Bell decoration ---
// Four gonad rings arranged in a clover (90° apart). Distance is the
// ring centre's radial offset from the bell axis in object space —
// inside this, RING_RADIUS sets the radius of each individual ring as
// it sits on the dome. Tuned against BELL_RADIUS = 0.32 so the rings
// land in the upper-mid portion of the dome, matching the gonad band
// of a real moon jellyfish.
const RING_DIST = 0.1;
const RING_RADIUS = 0.052;
const RING_THICKNESS = 0.018;

// Cosmetic upward thrust on each bell contraction. Doesn't drive
// long-range motion (the current does that) — just gives a visible
// "swim pulse" cadence to the height. Sized small relative to MAX_SPEED.
const PULSE_THRUST = 0.4;

// Maximum tilt angle the bell can take, in radians (~18°). The bell
// leans this much from vertical when the horizontal current is at or
// above TILT_SATURATION_SPEED — saturating prevents already-strong
// currents from tipping the jelly all the way onto its side.
const MAX_TILT = 0.32;
// Horizontal current speed at which tilt saturates to MAX_TILT.
// Currents below this scale the tilt linearly so weak eddies show as
// a subtle lean rather than a full tip.
const TILT_SATURATION_SPEED = 0.35;
// Exponential approach rate for the tilt vector — 1/TILT_RATE is the
// time constant. Lower = lazier reorientation; matches the slow,
// passive drift of a real jelly being pushed around.
const TILT_RATE = 0.9;

// Cap on the Y a jelly can integrate to. Surface is owned by Scene.
const CEIL_BUFFER = 0.4;

const _curr: [number, number, number] = [0, 0, 0];
const _steer: [number, number, number] = [0, 0, 0];
// Scratch axis for the per-instance quaternion build in the render
// pass — module scope so we don't allocate a Vector3 per jelly per
// frame.
const _tiltAxis = new THREE.Vector3();

function steerToward(
    dx: number,
    dy: number,
    dz: number,
    vx: number,
    vy: number,
    vz: number,
    maxSpeed: number,
): [number, number, number] {
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) {
        _steer[0] = 0;
        _steer[1] = 0;
        _steer[2] = 0;
        return _steer;
    }
    let sx = (dx / len) * maxSpeed - vx;
    let sy = (dy / len) * maxSpeed - vy;
    let sz = (dz / len) * maxSpeed - vz;
    const slen = Math.hypot(sx, sy, sz);
    if (slen > MAX_FORCE) {
        const c = MAX_FORCE / slen;
        sx *= c;
        sy *= c;
        sz *= c;
    }
    _steer[0] = sx;
    _steer[1] = sy;
    _steer[2] = sz;
    return _steer;
}

type Props = {
    surfaceY: number;
    count: number;
};

// Build a unit jellyfish: dome on top (centred at y=0, extending up
// to y=BELL_HEIGHT) plus TENTACLE_COUNT tapered tubes hanging below
// (down to y=-TENTACLE_LENGTH). Two per-vertex attributes drive the
// pulse shader:
//   aBellT     — 0 at rim, 1 at bell tip. Bell-only deformation mask.
//   aTentacleT — 0 at rim, 1 at tentacle tip. Tentacle undulation mask.
// A vertex on the bell has aTentacleT==0; a vertex on a tentacle has
// aBellT==0 and aTentacleT in [0, 1].
function buildJellyGeometry(): THREE.BufferGeometry {
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    const aBellT: number[] = [];
    const aTentacleT: number[] = [];
    // Per-vertex sway phase for tentacles, so each tentacle ripples on
    // its own offset and they don't undulate in unison.
    const aSwayPhase: number[] = [];

    // ---- Bell dome (upper hemisphere, vertically squashed) ----
    const bellStart = 0;
    for (let i = 0; i <= BELL_LAT_SEGMENTS; i++) {
        const v = i / BELL_LAT_SEGMENTS;
        const theta = v * Math.PI * 0.5;
        const y = Math.cos(theta) * BELL_HEIGHT;
        const r = Math.sin(theta) * BELL_RADIUS;
        // Bell mass concentrates at the top: bellT==1 at the dome tip,
        // 0 at the rim. The pulse swells the rim outward and squashes
        // the top down, so this is the right gradient.
        const bellT = 1 - v;
        for (let j = 0; j <= BELL_LON_SEGMENTS; j++) {
            const u = j / BELL_LON_SEGMENTS;
            const phi = u * Math.PI * 2;
            const x = Math.cos(phi) * r;
            const z = Math.sin(phi) * r;
            positions.push(x, y, z);
            // Outward + up normal — approximate; the dome is shallow so
            // this is close enough for soft lambert shading.
            const nx = Math.cos(phi) * Math.sin(theta);
            const ny = Math.cos(theta);
            const nz = Math.sin(phi) * Math.sin(theta);
            normals.push(nx, ny, nz);
            aBellT.push(bellT);
            aTentacleT.push(0);
            aSwayPhase.push(0);
        }
    }
    const lonStride = BELL_LON_SEGMENTS + 1;
    for (let i = 0; i < BELL_LAT_SEGMENTS; i++) {
        for (let j = 0; j < BELL_LON_SEGMENTS; j++) {
            const a = bellStart + i * lonStride + j;
            const b = a + 1;
            const c = a + lonStride;
            const d = c + 1;
            indices.push(a, c, b, b, c, d);
        }
    }

    // ---- Tentacles ----
    // Each tentacle is a short tapered prism with TENTACLE_SIDES
    // around its axis and TENTACLE_SEGMENTS along it. Hangs straight
    // down from a point on the bell rim; the shader bends the lower
    // segments to give the undulation.
    for (let t = 0; t < TENTACLE_COUNT; t++) {
        const angle = (t / TENTACLE_COUNT) * Math.PI * 2;
        // Attach slightly inside the rim so the tentacles read as
        // emerging from the bell rather than a separate disc.
        const baseX = Math.cos(angle) * BELL_RADIUS * 0.85;
        const baseZ = Math.sin(angle) * BELL_RADIUS * 0.85;
        // Phase offset per tentacle so they ripple out of unison.
        const swayPhase = (t / TENTACLE_COUNT) * Math.PI * 2;

        const ringStart: number[] = [];
        for (let s = 0; s <= TENTACLE_SEGMENTS; s++) {
            const ts = s / TENTACLE_SEGMENTS;
            const r =
                TENTACLE_BASE_RADIUS +
                (TENTACLE_TIP_RADIUS - TENTACLE_BASE_RADIUS) * ts;
            const y = -ts * TENTACLE_LENGTH;
            ringStart.push(positions.length / 3);
            for (let k = 0; k < TENTACLE_SIDES; k++) {
                const ang = (k / TENTACLE_SIDES) * Math.PI * 2;
                const ox = Math.cos(ang) * r;
                const oz = Math.sin(ang) * r;
                positions.push(baseX + ox, y, baseZ + oz);
                normals.push(Math.cos(ang), 0, Math.sin(ang));
                aBellT.push(0);
                aTentacleT.push(ts);
                aSwayPhase.push(swayPhase);
            }
        }
        // Connect successive rings around the tentacle.
        for (let s = 0; s < TENTACLE_SEGMENTS; s++) {
            const a = ringStart[s];
            const b = ringStart[s + 1];
            for (let k = 0; k < TENTACLE_SIDES; k++) {
                const k1 = (k + 1) % TENTACLE_SIDES;
                indices.push(a + k, a + k1, b + k);
                indices.push(b + k, a + k1, b + k1);
            }
        }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3),
    );
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geo.setAttribute("aBellT", new THREE.Float32BufferAttribute(aBellT, 1));
    geo.setAttribute(
        "aTentacleT",
        new THREE.Float32BufferAttribute(aTentacleT, 1),
    );
    geo.setAttribute(
        "aSwayPhase",
        new THREE.Float32BufferAttribute(aSwayPhase, 1),
    );
    geo.setIndex(indices);
    return geo;
}

const vertexShader = /* glsl */ `
    precision highp float;

    uniform float uTime;
    uniform float uPulseOmega;
    uniform float uBellRadialAmp;
    uniform float uBellVertAmp;
    uniform float uTentacleSway;
    // Surface Y in the parent group's local frame. Transformed by
    // modelMatrix here so the fragment shader has the surface plane
    // in world space, matching cameraPosition and vWorldPos.
    uniform float uSurfaceLocalY;

    attribute float aBellT;
    attribute float aTentacleT;
    attribute float aSwayPhase;
    attribute float aPhase;
    attribute vec3 aTint;

    varying vec3 vColor;
    varying vec3 vNormalW;
    varying vec3 vWorldPos;
    // World-space Y of the water surface plane. Constant per draw call,
    // so emitting from the vertex shader is just a way to apply the
    // group transform once without plumbing it through JS.
    varying float vWaterY;
    // Undeformed object-space XZ of this vertex. Used in the fragment
    // shader to evaluate the 4-ring pattern in a stable coordinate
    // frame — sampling the pulsing post-deformation XZ would make the
    // rings expand and contract with the bell, which they shouldn't.
    varying vec2 vOrigXZ;
    varying float vBellT;
    varying float vTentacleT;

    void main() {
        float phase = uTime * uPulseOmega + aPhase;
        float pulse = sin(phase);

        vec3 pos = position;
        vec2 origXZ = position.xz;

        // Bell pulse. The rim (aBellT == 0) swells outward and the tip
        // (aBellT == 1) presses down — same as a real moon jelly's
        // contraction: tip-down, rim-out → push water down → thrust up.
        // We're rendering the deformation; the world-space thrust is
        // applied in the JS integrator.
        if (aBellT > 0.0001) {
            float rimMask = 1.0 - aBellT;
            float radial = pulse * uBellRadialAmp * rimMask;
            pos.x *= 1.0 + radial;
            pos.z *= 1.0 + radial;
            pos.y -= pulse * uBellVertAmp * aBellT;
        }

        // Tentacle sway. Lags the bell by ~π/2 so tentacles trail
        // through the pulse rather than moving in lockstep. Amplitude
        // ramps with aTentacleT so the rim stays anchored and the tip
        // swings.
        if (aTentacleT > 0.0001) {
            float lag = sin(phase - 1.5708 + aSwayPhase + aTentacleT * 3.0);
            float amp = uTentacleSway * aTentacleT;
            pos.x += cos(aSwayPhase) * lag * amp;
            pos.z += sin(aSwayPhase) * lag * amp;
        }

        vec4 worldPos = modelMatrix * instanceMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * viewMatrix * worldPos;

        // World-space normal so the fragment can do a stable Fresnel
        // against the world-space view ray. Going through normalMatrix
        // here would re-project into view space, which fights the
        // cameraPosition lookup below.
        vec3 instanceNormal = mat3(instanceMatrix) * normal;
        vNormalW = normalize(mat3(modelMatrix) * instanceNormal);
        vWorldPos = worldPos.xyz;
        vWaterY = (modelMatrix * vec4(0.0, uSurfaceLocalY, 0.0, 1.0)).y;
        vColor = aTint;
        vOrigXZ = origXZ;
        vBellT = aBellT;
        vTentacleT = aTentacleT;
    }
`;

const fragmentShader = /* glsl */ `
    precision highp float;

    uniform vec3 uRimColor;
    uniform vec3 uRingColor;
    uniform float uRingDist;
    uniform float uRingRadius;
    uniform float uRingThickness;
    // Beer–Lambert parameters, matched to Water.tsx so the jelly
    // self-tints consistently with the volumetric water shader. The
    // jelly doesn't write depth, so the water shader can't tint it via
    // its depth-pre-pass; applying the same extinction here means the
    // jelly picks up the same chromatic shift it would have if it
    // *had* contributed to the depth pre-pass.
    uniform vec3 uWaterTint;
    uniform vec3 uExtinction;
    uniform float uMaxPathLength;

    varying vec3 vColor;
    varying vec3 vNormalW;
    varying vec3 vWorldPos;
    varying float vWaterY;
    varying vec2 vOrigXZ;
    varying float vBellT;
    varying float vTentacleT;

    const float PI = 3.1415926535;

    // Length of the camera→fragment ray that lies below the water
    // surface plane. If the camera is above and the fragment is below
    // (the typical case for this scene), only the underwater portion
    // of the ray gets counted; an entirely-above-water fragment counts
    // zero. Capped at uMaxPathLength so a far-away jelly viewed across
    // the entire tank doesn't saturate to pure tint.
    float underwaterPathLength() {
        float camY = cameraPosition.y;
        float fragY = vWorldPos.y;
        float total = distance(cameraPosition, vWorldPos);
        if (fragY >= vWaterY) return 0.0;
        if (camY <= vWaterY) return min(total, uMaxPathLength);
        // Camera above, fragment below. Find ray parameter at the
        // water-surface crossing and count only the segment past it.
        float tCross = (vWaterY - camY) / (fragY - camY);
        float waterDist = (1.0 - tCross) * total;
        return min(waterDist, uMaxPathLength);
    }

    // Standard Beer–Lambert transmission. Returns per-channel
    // transmittance T ∈ [0,1]: T=1 means no attenuation (no path),
    // T→0 means fully extinguished (very long path). Red dies fastest
    // under uExtinction so the residual tint shifts cyan with depth.
    vec3 waterTransmittance(float d) {
        return exp(-uExtinction * d);
    }

    void main() {
        vec3 N = normalize(vNormalW);
        // cameraPosition is a built-in uniform on ShaderMaterial.
        vec3 V = normalize(cameraPosition - vWorldPos);
        float ndotv = abs(dot(N, V));
        // Fresnel factor: 0 looking straight at the surface, 1 at
        // grazing. Drives both the rim glow and the alpha — face-on
        // pixels are see-through, edge pixels are bright.
        float fresnel = 1.0 - ndotv;

        vec3 lightDir = normalize(vec3(10.0, 15.0, 8.0));
        float ndotl = abs(dot(N, lightDir));

        // Pre-compute the underwater path length once; both branches
        // below need it to apply the chromatic tint.
        float pathLen = underwaterPathLength();
        vec3 T = waterTransmittance(pathLen);

        if (vBellT > 0.0001) {
            // ---- Four-ring gonad pattern ----
            // The four gonad horseshoes of a moon jelly sit roughly at
            // 90° intervals around the bell axis, just inside the rim.
            // Find the distance from this fragment's undeformed XZ to
            // the nearest of four centre points placed on a circle of
            // radius uRingDist; the ring band is where that distance
            // equals uRingRadius (annular mask).
            float minD = 1e6;
            for (int k = 0; k < 4; k++) {
                float a = float(k) * (PI * 0.5);
                vec2 c = vec2(cos(a), sin(a)) * uRingDist;
                float d = length(vOrigXZ - c);
                minD = min(minD, d);
            }
            float ring = 1.0 -
                smoothstep(0.0, uRingThickness, abs(minD - uRingRadius));
            // Suppress rings on the lower bell (near the rim, where
            // bellT is small) — the gonads are central in real moon
            // jellies, not at the edge.
            float ringEnable = smoothstep(0.1, 0.4, vBellT);
            ring *= ringEnable;

            // ---- Radial canal striations ----
            // Thin bright lines running rim → tip, evoking the radial
            // canal system visible through a real moon jelly's bell.
            // sin(theta * N) creates N peaks around the dome; raising
            // to a high power narrows each peak into a hair-fine line.
            float theta = atan(vOrigXZ.y, vOrigXZ.x);
            float stri = abs(sin(theta * 9.0));
            stri = pow(stri, 18.0);
            // Fade the lines out at the rim and at the very top — they
            // run through the central band of the dome.
            stri *= smoothstep(0.04, 0.35, vBellT) *
                    (1.0 - smoothstep(0.75, 0.98, vBellT));

            // ---- Composite colour ----
            // Soft lambert + generous ambient — translucent bodies
            // read brighter than the diffuse-only lighting suggests
            // because they transmit + scatter light from the back.
            float lit = 0.7 + 0.4 * ndotl;
            vec3 base = vColor * lit;
            // Rim glow: shift toward uRimColor at grazing angles.
            base = mix(base, uRimColor, pow(fresnel, 2.5));
            // Ring brightening — punch the rings toward uRingColor so
            // they read clearly through the body tint.
            base = mix(base, uRingColor, ring * 0.7);
            // Striations add a faint additive ridge — too subtle to
            // dominate, but enough to break up the smooth dome.
            base += uRimColor * stri * 0.2;

            // ---- Translucent alpha ----
            // Face-on pixels: low alpha (see-through). Grazing: high
            // alpha (rim glow). Rings and striations override toward
            // higher alpha so they don't fade away in the middle of
            // the bell.
            float alpha = mix(0.32, 0.88, pow(fresnel, 1.6));
            alpha = max(alpha, ring * 0.92);
            alpha = max(alpha, stri * 0.7);

            // Beer–Lambert: extinguish the jelly's emitted colour and
            // blend the absorbed light back toward uWaterTint. Same
            // form as the water shader's fragment output, so a jelly
            // at depth d carries the same chromatic shift as the rest
            // of the scene viewed through d units of water — no more
            // "white blob hovering over a blue scene" disconnect.
            base = base * T + uWaterTint * (1.0 - T);

            gl_FragColor = vec4(base, alpha);
        } else {
            // ---- Tentacles ----
            // Slightly soft, taking a hint of rim colour at the edges
            // for cohesion with the bell; opacity fades toward the
            // tips for the wispy look.
            float lit = 0.7 + 0.35 * ndotl;
            vec3 base = vColor * lit;
            base = mix(base, uRimColor, pow(fresnel, 2.0) * 0.45);
            float alpha = mix(0.85, 0.32, vTentacleT);
            // Same self-tint as the bell so the tentacles don't read
            // as warmer than the body they hang off.
            base = base * T + uWaterTint * (1.0 - T);
            gl_FragColor = vec4(base, alpha);
        }
    }
`;

// Near-pure white with the faintest cool cast. Pushed up from the
// earlier #f1eef6 lavender so the bells read brighter against the
// cyan water tint — the Beer–Lambert self-tint multiplies this colour
// before output, so starting brighter is the cheapest way to claw
// back contrast without dialling the tint itself down.
const BASE_COLOR = new THREE.Color("#fafcff");
const HUE_JITTER = 0.04;
const SAT_JITTER = 0.08;
const LIGHT_JITTER = 0.06;

export default function Jellyfish({ surfaceY, count }: Props) {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    const yMax = surfaceY - CEIL_BUFFER;
    const yFloor = -1.0;

    // Per-jellyfish state in noise-world coords. Same SoA pattern as
    // Fish — Float32Arrays touched by the per-frame loop with no
    // allocations. `phase` is the per-instance pulse offset so the
    // population doesn't pulse in unison.
    const state = useMemo(() => {
        const pos = new Float32Array(count * 3);
        const vel = new Float32Array(count * 3);
        const phase = new Float32Array(count);
        // Per-jelly tilt vector (tiltX, tiltZ) in the horizontal plane.
        // Magnitude is the tilt angle in radians; direction is where
        // the bell axis leans toward. Stored (rather than recomputed
        // from velocity) so we can lazily lerp toward the target — the
        // current direction shifts faster than a passive drifter
        // should physically reorient.
        const tilt = new Float32Array(count * 2);
        const spawnHalf = XZ_BOUND - XZ_MARGIN;
        const ceilY = yMax - 0.3;
        for (let i = 0; i < count; i++) {
            let x = 0,
                z = 0,
                terrainH = 0;
            for (let attempt = 0; attempt < 4; attempt++) {
                x = (Math.random() - 0.5) * 2 * spawnHalf;
                z = (Math.random() - 0.5) * 2 * spawnHalf;
                terrainH = sampleTerrain(x, z).height;
                if (terrainH < yMax - 1.5) break;
            }
            const floorY = terrainH + TERRAIN_CLEARANCE + 0.2;
            const y =
                floorY < ceilY
                    ? floorY + Math.random() * (ceilY - floorY)
                    : ceilY - 0.05;
            pos[i * 3 + 0] = x;
            pos[i * 3 + 1] = y;
            pos[i * 3 + 2] = z;
            // Start near rest — the current will pick them up on its
            // own within a frame or two and a non-zero start would just
            // make the first second look jittery.
            vel[i * 3 + 0] = (Math.random() - 0.5) * 0.1;
            vel[i * 3 + 1] = 0;
            vel[i * 3 + 2] = (Math.random() - 0.5) * 0.1;
            phase[i] = Math.random() * Math.PI * 2;
            // tilt starts at (0, 0) — bell upright until the current
            // pushes it into a lean.
        }
        return { pos, vel, phase, tilt };
    }, [yMax, count]);

    // Bake the unit jelly geometry once. Per-instance phase and tint
    // attributes are added below — same pattern as Seagrass / Coral.
    const geometry = useMemo(() => {
        const g = buildJellyGeometry();
        const phaseAttr = new THREE.InstancedBufferAttribute(
            new Float32Array(count),
            1,
        );
        g.setAttribute("aPhase", phaseAttr);
        const tintAttr = new THREE.InstancedBufferAttribute(
            new Float32Array(count * 3),
            3,
        );
        g.setAttribute("aTint", tintAttr);

        // Fill the per-instance buffers once — phase is sampled the same
        // way the integrator uses it, and tint is just cosmetic jitter
        // that doesn't need to change at runtime.
        const phaseArr = phaseAttr.array as Float32Array;
        const tintArr = tintAttr.array as Float32Array;
        const hsl = { h: 0, s: 0, l: 0 };
        BASE_COLOR.getHSL(hsl);
        const tmp = new THREE.Color();
        for (let i = 0; i < count; i++) {
            phaseArr[i] = state.phase[i];
            const h2 = hsl.h + (Math.random() - 0.5) * HUE_JITTER;
            const s2 = Math.max(
                0,
                Math.min(1, hsl.s + (Math.random() - 0.5) * SAT_JITTER),
            );
            const l2 = Math.max(
                0,
                Math.min(1, hsl.l + (Math.random() - 0.5) * LIGHT_JITTER),
            );
            tmp.setHSL(h2, s2, l2);
            tintArr[i * 3 + 0] = tmp.r;
            tintArr[i * 3 + 1] = tmp.g;
            tintArr[i * 3 + 2] = tmp.b;
        }
        return g;
    }, [count, state]);

    const uniforms = useMemo(
        () => ({
            uTime: { value: 0 },
            uPulseOmega: { value: PULSE_OMEGA },
            uBellRadialAmp: { value: BELL_RADIAL_AMP },
            uBellVertAmp: { value: BELL_VERT_AMP },
            uTentacleSway: { value: TENTACLE_SWAY },
            // Cool near-white for the grazing-angle glow. Reads as
            // "rim" through the cyan water tint without going so blue
            // that it disappears against the background.
            uRimColor: { value: new THREE.Color("#e6f1ff") },
            // Brighter, slightly warm white for the gonad rings —
            // matches the way a real moon jelly's gonads pick up light
            // from the body's interior scattering.
            uRingColor: { value: new THREE.Color("#ffffff") },
            uRingDist: { value: RING_DIST },
            uRingRadius: { value: RING_RADIUS },
            uRingThickness: { value: RING_THICKNESS },
            // Surface plane Y in the parent group's local frame.
            // Vertex shader applies modelMatrix to get the world-space
            // surface used by the underwater-path-length calculation.
            uSurfaceLocalY: { value: surfaceY },
            // Water tint + extinction must match Water.tsx so the
            // jelly's self-tint blends seamlessly with the volumetric
            // water rendering. If those constants change there, mirror
            // them here.
            uWaterTint: { value: new THREE.Color("#42BEBE") },
            uExtinction: { value: new THREE.Vector3(0.55, 0.25, 0.18) },
            uMaxPathLength: { value: 20.0 },
        }),
        [surfaceY],
    );

    // Absolute wall-clock seconds the current field samples against —
    // same value the shader reads, so the integrator's pulse thrust
    // stays in phase with the visible bell contraction.
    const timeRef = useRef(0);

    useFrame((_, delta) => {
        const mesh = meshRef.current;
        if (!mesh) return;
        const dt = Math.min(delta, 0.05);
        timeRef.current += dt;
        uniforms.uTime.value = timeRef.current;

        const { pos, vel, phase, tilt } = state;
        const t = timeRef.current;
        const sepSq = SEPARATION * SEPARATION;
        const percSq = PERCEPTION * PERCEPTION;

        // O(N²) separation only — jellies don't align or follow each
        // other, so we don't need the spatial-hash machinery the fish
        // use. At expected counts (≤ ~60) this is comfortably cheap;
        // if the population grew much larger this would be the first
        // thing to swap for a uniform grid.
        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            const px = pos[ix],
                py = pos[ix + 1],
                pz = pos[ix + 2];
            const vx = vel[ix],
                vy = vel[ix + 1],
                vz = vel[ix + 2];

            let sepX = 0,
                sepY = 0,
                sepZ = 0;
            let separators = 0;
            for (let j = 0; j < count; j++) {
                if (j === i) continue;
                const jx = j * 3;
                const dx = pos[jx] - px;
                const dy = pos[jx + 1] - py;
                const dz = pos[jx + 2] - pz;
                const distSq = dx * dx + dy * dy + dz * dz;
                if (distSq < percSq && distSq < sepSq && distSq > 1e-6) {
                    const inv = 1.0 / Math.sqrt(distSq);
                    sepX -= dx * inv;
                    sepY -= dy * inv;
                    sepZ -= dz * inv;
                    separators++;
                }
            }

            let ax = 0,
                ay = 0,
                az = 0;

            // ---- Current advection ----
            // Steer the velocity toward the local current vector. With
            // a high weight, the current essentially sets the velocity
            // within a fraction of a second; lower weights would let
            // jellies feel "draggy" and lag the field.
            sampleCurrent(px, py, pz, t, _curr);
            const cs = steerToward(
                _curr[0],
                _curr[1],
                _curr[2],
                vx,
                vy,
                vz,
                MAX_SPEED,
            );
            ax += cs[0] * W_CURRENT;
            ay += cs[1] * W_CURRENT;
            az += cs[2] * W_CURRENT;

            // ---- Tilt toward current ----
            // The bell axis (local +Y) leans toward the horizontal
            // current direction — same response a real jelly's body
            // shows when pushed by flow. The stored tilt is a 2D
            // vector in the horizontal plane: its direction is where
            // +Y leans toward, its magnitude is the lean angle in
            // radians. Target magnitude saturates at MAX_TILT so a
            // strong gyre doesn't tip the jelly fully onto its side;
            // exponential lerp toward target so the response is lazy.
            const cxh = _curr[0];
            const czh = _curr[2];
            const horizSpeed = Math.hypot(cxh, czh);
            let targetTX = 0;
            let targetTZ = 0;
            if (horizSpeed > 1e-4) {
                const tiltMag =
                    Math.min(horizSpeed / TILT_SATURATION_SPEED, 1) * MAX_TILT;
                targetTX = (cxh / horizSpeed) * tiltMag;
                targetTZ = (czh / horizSpeed) * tiltMag;
            }
            const ti = i * 2;
            const lerpAmt = Math.min(1, TILT_RATE * dt);
            tilt[ti] += (targetTX - tilt[ti]) * lerpAmt;
            tilt[ti + 1] += (targetTZ - tilt[ti + 1]) * lerpAmt;

            // ---- Loose separation ----
            if (separators > 0) {
                const s = steerToward(sepX, sepY, sepZ, vx, vy, vz, MAX_SPEED);
                ax += s[0] * W_SEPARATION;
                ay += s[1] * W_SEPARATION;
                az += s[2] * W_SEPARATION;
            }

            // ---- Soft XZ walls ----
            if (px > XZ_BOUND - XZ_MARGIN)
                ax -= (px - (XZ_BOUND - XZ_MARGIN)) * W_BOUNDS;
            if (px < -XZ_BOUND + XZ_MARGIN)
                ax += (-XZ_BOUND + XZ_MARGIN - px) * W_BOUNDS;
            if (pz > XZ_BOUND - XZ_MARGIN)
                az -= (pz - (XZ_BOUND - XZ_MARGIN)) * W_BOUNDS;
            if (pz < -XZ_BOUND + XZ_MARGIN)
                az += (-XZ_BOUND + XZ_MARGIN - pz) * W_BOUNDS;
            // Soft surface ceiling. Bell tip mustn't breach.
            if (py > yMax - 0.3) ay -= (py - (yMax - 0.3)) * W_BOUNDS * 2.0;

            // ---- Terrain avoidance ----
            const speedXZ = Math.hypot(vx, vz);
            let aheadX = px;
            let aheadZ = pz;
            if (speedXZ > 1e-3) {
                aheadX = px + (vx / speedXZ) * TERRAIN_LOOKAHEAD;
                aheadZ = pz + (vz / speedXZ) * TERRAIN_LOOKAHEAD;
            }
            const gHere = sampleTerrain(px, pz).height;
            const gAhead = sampleTerrain(aheadX, aheadZ).height;
            const ceiling = Math.max(gHere, gAhead) + TERRAIN_CLEARANCE;
            if (py < ceiling) {
                const overlap = ceiling - py;
                ay += overlap * W_TERRAIN_UP;
                const gPlusX = sampleTerrain(px + TERRAIN_GRAD_EPS, pz).height;
                const gPlusZ = sampleTerrain(px, pz + TERRAIN_GRAD_EPS).height;
                const gradX = (gPlusX - gHere) / TERRAIN_GRAD_EPS;
                const gradZ = (gPlusZ - gHere) / TERRAIN_GRAD_EPS;
                if (gradX * gradX + gradZ * gradZ > 1e-6) {
                    const s = steerToward(
                        -gradX,
                        0,
                        -gradZ,
                        vx,
                        0,
                        vz,
                        MAX_SPEED,
                    );
                    ax += s[0] * W_TERRAIN_REDIRECT;
                    az += s[2] * W_TERRAIN_REDIRECT;
                }
            }

            // ---- Pulse thrust ----
            // Cosmetic: each bell contraction (positive pulse) gives a
            // small upward kick, so the height oscillates in time with
            // the visible bell animation. Squared so only the active
            // half of the cycle pushes; the relaxing half drifts.
            const ph = t * PULSE_OMEGA + phase[i];
            const pulse = Math.sin(ph);
            const thrust = pulse > 0 ? pulse * pulse * PULSE_THRUST : 0;
            ay += thrust;

            // Integrate.
            let nvx = vx + ax * dt;
            let nvy = vy + ay * dt;
            let nvz = vz + az * dt;
            const speed = Math.hypot(nvx, nvy, nvz);
            if (speed > MAX_SPEED) {
                const c = MAX_SPEED / speed;
                nvx *= c;
                nvy *= c;
                nvz *= c;
            } else if (speed < MIN_SPEED && speed > 1e-6) {
                const c = MIN_SPEED / speed;
                nvx *= c;
                nvy *= c;
                nvz *= c;
            }
            vel[ix] = nvx;
            vel[ix + 1] = nvy;
            vel[ix + 2] = nvz;

            pos[ix] += nvx * dt;
            pos[ix + 1] += nvy * dt;
            pos[ix + 2] += nvz * dt;

            // Hard clamps.
            if (pos[ix] > XZ_BOUND) pos[ix] = XZ_BOUND;
            if (pos[ix] < -XZ_BOUND) pos[ix] = -XZ_BOUND;
            if (pos[ix + 2] > XZ_BOUND) pos[ix + 2] = XZ_BOUND;
            if (pos[ix + 2] < -XZ_BOUND) pos[ix + 2] = -XZ_BOUND;
            if (pos[ix + 1] > yMax) pos[ix + 1] = yMax;
            if (pos[ix + 1] < yFloor) pos[ix + 1] = yFloor;
        }

        // ---- Render pass ----
        // World → scene-local subtraction so the slice rendering shows
        // the local window of the world. Jellies stay upright (no roll
        // or pitch with motion), so the instance matrix is just a
        // translation — no lookAt is needed and the bell axis stays
        // aligned with world Y, which matches how moon jellies behave.
        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            const lx = pos[ix] - worldOffset.x;
            const ly = pos[ix + 1];
            const lz = pos[ix + 2] - worldOffset.z;
            // Cull jellies far outside the visible slice by parking
            // them at a degenerate scale. Cheaper than rebuilding the
            // buffer; keeps the draw count constant.
            const half = WORLD_HALF_EXTENT;
            if (
                lx < -half * 1.2 ||
                lx > half * 1.2 ||
                lz < -half * 1.2 ||
                lz > half * 1.2
            ) {
                dummy.scale.setScalar(0);
            } else {
                dummy.scale.setScalar(1);
            }
            // Build the lean rotation directly from the stored tilt
            // vector. The axis perpendicular to both +Y and the tilt
            // direction is Y × tiltDir = (tz, 0, -tx); rotating by the
            // tilt magnitude around that axis takes +Y over to the
            // tilt direction. Stays as a quaternion so updateMatrix
            // doesn't have to round-trip through Euler.
            const ti = i * 2;
            const tx = tilt[ti];
            const tz = tilt[ti + 1];
            const tiltMag = Math.hypot(tx, tz);
            if (tiltMag > 1e-5) {
                _tiltAxis.set(tz / tiltMag, 0, -tx / tiltMag);
                dummy.quaternion.setFromAxisAngle(_tiltAxis, tiltMag);
            } else {
                dummy.quaternion.identity();
            }
            dummy.position.set(lx, ly, lz);
            dummy.updateMatrix();
            meshRef.current!.setMatrixAt(i, dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh
            ref={meshRef}
            key={count}
            args={[geometry, undefined, count]}
            frustumCulled={false}
        >
            <shaderMaterial
                uniforms={uniforms}
                vertexShader={vertexShader}
                fragmentShader={fragmentShader}
                side={THREE.DoubleSide}
                // Translucent bell. depthWrite off so the back of the
                // bell shows through the front and so jellies behind
                // each other blend instead of punching out. depthTest
                // stays on so opaque scene elements (terrain, fish)
                // still occlude jellies behind them.
                transparent={true}
                depthWrite={false}
            />
        </instancedMesh>
    );
}
