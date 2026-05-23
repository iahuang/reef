import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { SEED, fbm, sampleTerrain, smoothstep } from "./terrain";
import { worldOffset } from "./worldOffset";

type Props = {
    size: number;
    waterY: number;
    variantCount?: number;
};

const DEFAULT_VARIANT_COUNT = 4;

// Placement — brain corals are boulder-form, so they live on flat low
// ground: the basin floor and the lower shelf, not the rising reef.
// CELL_SIZE larger than branching coral's since each mound occupies a
// wider footprint and they shouldn't carpet the floor.
const CELL_SIZE = 1.8;
const PLACEMENT_STRATUM_MIN = 0.3;
const PLACEMENT_STRATUM_MAX = 0.55;

// Patch mask. Distinct seed from Seagrass (99) and Coral (199) so the
// colonies aren't co-located on the same noise contours.
const PATCH_FREQUENCY = 0.16;
const PATCH_THRESHOLD = 0.52;
const PATCH_EDGE_WIDTH = 0.1;
const PATCH_SEED_OFFSET = 311;

const VARIANT_SEED_OFFSET = 2000;

const SUBMERGE_BUFFER = 0.1;

const SIZE_MIN = 0.6;
const SIZE_MAX = 1.4;

// Muted earthen palette — brain corals read calmer/browner than the
// vivid branching variants, which keeps the seabed from looking like a
// rainbow.
const BRAIN_PALETTE = [
    new THREE.Color("#c8a67a"),
    new THREE.Color("#a07a4c"),
    new THREE.Color("#8b9a5b"),
    new THREE.Color("#b86e54"),
    new THREE.Color("#daa57a"),
    new THREE.Color("#705c43"),
];
const HUE_JITTER = 0.04;
const SAT_JITTER = 0.12;
const LIGHT_JITTER = 0.15;

// Geometry: start from an icosphere, push each vertex outward by 3D
// fBM (gives the irregular boulder shape), then squash Y to mound
// proportions. Subdivisions=2 gives 320 tris — coarse enough for the
// pixel-art look (flat-shaded), fine enough for the surface to read as
// curved rather than polyhedral.
const ICO_SUBDIVISIONS = 2;
const BASE_RADIUS = 0.4;
const FLATTEN_Y = 0.5;
// Shift the mound up so most of it sits above the terrain instead of
// the centred sphere being half-buried.
const Y_SHIFT = BASE_RADIUS * FLATTEN_Y * 0.5;
const DISPLACE_AMPLITUDE = 0.12;
const DISPLACE_FREQUENCY = 1.8;

// Same per-cell hash family as Seagrass / Coral.
function cellHash(cx: number, cz: number, channel: number): number {
    const s =
        Math.sin(cx * 127.1 + cz * 311.7 + channel * 53.7) * 43758.5453;
    return s - Math.floor(s);
}

// 3D value-noise family — terrain.ts only exports a 2D one, and the
// brain-coral displacement + the in-shader groove pattern both need
// stable 3D noise. Kept local since no other component needs it.
function hash3D(x: number, y: number, z: number, seed: number): number {
    const s =
        Math.sin(x * 127.1 + y * 311.7 + z * 74.7 + seed * 0.013) *
        43758.5453;
    return s - Math.floor(s);
}

function valueNoise3D(
    x: number,
    y: number,
    z: number,
    seed: number,
): number {
    const ix = Math.floor(x);
    const iy = Math.floor(y);
    const iz = Math.floor(z);
    const fx = x - ix;
    const fy = y - iy;
    const fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const uz = fz * fz * (3 - 2 * fz);
    const v000 = hash3D(ix, iy, iz, seed);
    const v100 = hash3D(ix + 1, iy, iz, seed);
    const v010 = hash3D(ix, iy + 1, iz, seed);
    const v110 = hash3D(ix + 1, iy + 1, iz, seed);
    const v001 = hash3D(ix, iy, iz + 1, seed);
    const v101 = hash3D(ix + 1, iy, iz + 1, seed);
    const v011 = hash3D(ix, iy + 1, iz + 1, seed);
    const v111 = hash3D(ix + 1, iy + 1, iz + 1, seed);
    const x00 = v000 * (1 - ux) + v100 * ux;
    const x10 = v010 * (1 - ux) + v110 * ux;
    const x01 = v001 * (1 - ux) + v101 * ux;
    const x11 = v011 * (1 - ux) + v111 * ux;
    const y0 = x00 * (1 - uy) + x10 * uy;
    const y1 = x01 * (1 - uy) + x11 * uy;
    return y0 * (1 - uz) + y1 * uz;
}

function fbm3D(x: number, y: number, z: number, seed: number): number {
    let v = 0;
    let amp = 0.5;
    let freq = 1;
    for (let i = 0; i < 3; i++) {
        v += amp * valueNoise3D(x * freq, y * freq, z * freq, seed + i * 17);
        amp *= 0.5;
        freq *= 2;
    }
    return v;
}

// Bake one mound variant. Each vertex on the base icosphere is pushed
// outward along its radial direction by 3D fBM, then the whole shape is
// flattened vertically and shifted up. computeVertexNormals after gives
// per-face normals (the geometry is non-indexed, which produces flat
// shading — fits the scene's pixel-art aesthetic).
function buildVariant(
    seed: number,
): { geometry: THREE.BufferGeometry; maxY: number } {
    const geo = new THREE.IcosahedronGeometry(BASE_RADIUS, ICO_SUBDIVISIONS);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;
    let maxY = 0;
    for (let i = 0; i < arr.length; i += 3) {
        const x0 = arr[i];
        const y0 = arr[i + 1];
        const z0 = arr[i + 2];
        const len = Math.sqrt(x0 * x0 + y0 * y0 + z0 * z0) || 1;
        // Sample displacement at the unsquashed sphere position so the
        // bumps are isotropic — sampling after the Y squash would
        // stretch features vertically.
        const n = fbm3D(
            x0 * DISPLACE_FREQUENCY,
            y0 * DISPLACE_FREQUENCY,
            z0 * DISPLACE_FREQUENCY,
            seed,
        );
        const d = (n - 0.5) * 2 * DISPLACE_AMPLITUDE;
        const r = len + d;
        const nx = x0 / len;
        const ny = y0 / len;
        const nz = z0 / len;
        const x = r * nx;
        const y = r * ny * FLATTEN_Y + Y_SHIFT;
        const z = r * nz;
        arr[i] = x;
        arr[i + 1] = y;
        arr[i + 2] = z;
        if (y > maxY) maxY = y;
    }
    geo.computeVertexNormals();
    return { geometry: geo, maxY };
}

const vertexShader = /* glsl */ `
    precision highp float;

    attribute vec3 aTint;
    attribute float aSeed;

    varying vec3 vColor;
    varying vec3 vNormal;
    varying vec3 vObjectPos;
    varying float vSeed;

    void main() {
        vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
        vec3 instanceNormal = mat3(instanceMatrix) * normal;
        vNormal = normalize(normalMatrix * instanceNormal);
        vColor = aTint;
        // Object-space position drives the surface groove sampling so
        // the pattern stays glued to the mesh as the slice scrolls.
        vObjectPos = position;
        vSeed = aSeed;
    }
`;

const fragmentShader = /* glsl */ `
    precision highp float;

    varying vec3 vColor;
    varying vec3 vNormal;
    varying vec3 vObjectPos;
    varying float vSeed;

    // Cheap 3D fBM in shader for the groove pattern. Three octaves keep
    // fragment cost low while still producing usable warp detail.
    float hash3(vec3 p) {
        return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
    }

    float vnoise3(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        return mix(
            mix(
                mix(hash3(i + vec3(0.0, 0.0, 0.0)), hash3(i + vec3(1.0, 0.0, 0.0)), f.x),
                mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), f.x),
                f.y),
            mix(
                mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), f.x),
                mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), f.x),
                f.y),
            f.z);
    }

    float fbm3(vec3 p) {
        float v = 0.0;
        float a = 0.5;
        for (int i = 0; i < 3; i++) {
            v += a * vnoise3(p);
            p *= 2.0;
            a *= 0.5;
        }
        return v;
    }

    // Tunable parameters.
    // STRIPE_FREQ — ridge periods per unit of warpedField output.
    //              7-10 gives a brain-density pattern.
    // PATTERN_SCALE — spatial frequency of the warp field input. Bigger
    //              = smaller meanders; smaller = sweeping curls.
    // WARP_AMOUNT — strength of the iterated displacement. Iquilez
    //              uses ~4 for strong labyrinthine field. Higher than
    //              that and the field becomes too chaotic.
    // RELIEF_STR — bump perturbation strength. Higher = deeper-looking
    //              valleys. Can go high (0.5+) because the finite-
    //              difference gradient is reliable at any DPR.
    // VALLEY_DK — extra valley shadow on top of bumped lambert.
    const float STRIPE_FREQ = 7.0;
    const float PATTERN_SCALE = 5.0;
    const float WARP_AMOUNT = 4.0;
    const float RELIEF_STR = 0.55;
    const float VALLEY_DK = 0.45;

    // Iquilez iterated domain warp. Inner fbm produces a vector that
    // displaces the outer fbm sample position — the resulting scalar
    // field is non-monotonic and chaotic, so taking iso-contours of it
    // gives ridges that fork, merge, and meander like real brain coral
    // rather than running in parallel bands.
    float warpedField(vec3 p) {
        vec3 q = vec3(
            fbm3(p),
            fbm3(p + vec3(5.2, 1.3, 8.7)),
            fbm3(p + vec3(2.8, 5.5, 1.1))
        );
        return fbm3(p + WARP_AMOUNT * q);
    }

    // Stripe phase. sin of warpedField output gives STRIPE_FREQ ridge
    // contours per unit of field range; since the field is smooth and
    // varied, contours adopt orientations that change across the
    // surface — locally parallel, globally labyrinthine.
    float stripePhase(vec3 p) {
        float n = warpedField(p * PATTERN_SCALE);
        return sin(n * STRIPE_FREQ * 3.14159);
    }

    void main() {
        // Per-instance offset → distinct stripe topology per mound.
        vec3 p = vObjectPos + vec3(vSeed * 17.0);
        float n = warpedField(p * PATTERN_SCALE);
        float angle = n * STRIPE_FREQ * 3.14159;
        float h = sin(angle);

        // Object-space finite-difference gradient of the SMOOTH warped
        // field (not the high-frequency sin output). Then apply the
        // analytic chain rule for sin to get the gradient of h. This
        // is reliable regardless of fragment density, unlike dFdx/dFdy
        // on a high-frequency sin which aliases at low DPR.
        const float EPS = 0.004;
        float nx = warpedField((p + vec3(EPS, 0.0, 0.0)) * PATTERN_SCALE);
        float ny = warpedField((p + vec3(0.0, EPS, 0.0)) * PATTERN_SCALE);
        float nz = warpedField((p + vec3(0.0, 0.0, EPS)) * PATTERN_SCALE);
        vec3 gradN = vec3(nx - n, ny - n, nz - n) / EPS;
        vec3 gradH = cos(angle) * STRIPE_FREQ * 3.14159 * gradN;

        // Project onto the surface tangent plane so the perturbation
        // moves the normal along the surface, not into/out of it.
        vec3 N = normalize(vNormal);
        gradH -= N * dot(gradH, N);
        vec3 Np = normalize(N - gradH * RELIEF_STR);

        // Lambert on the bumped normal. max() (not abs()) so the dark
        // side of every micro-ridge actually goes dark — that contrast
        // is what sells the depth illusion.
        vec3 lightDir = normalize(vec3(10.0, 15.0, 8.0));
        float ndotl = max(0.0, dot(Np, lightDir));
        float lit = 0.35 + 0.65 * ndotl;

        // Push valley contrast beyond the bumped lambert.
        float valley = smoothstep(0.2, -0.6, h);
        vec3 baseColor = vColor * (1.0 - valley * VALLEY_DK);

        gl_FragColor = vec4(baseColor * lit, 1.0);
    }
`;

export default function BrainCoral({
    size,
    waterY,
    variantCount = DEFAULT_VARIANT_COUNT,
}: Props) {
    const targetCells = useMemo(() => {
        const cellsPerSide = Math.ceil(size / CELL_SIZE) + 3;
        return cellsPerSide * cellsPerSide;
    }, [size]);

    const perVariantCapacity = useMemo(
        () => Math.ceil((targetCells * 2) / variantCount) + 16,
        [targetCells, variantCount],
    );

    const variants = useMemo(() => {
        const out: {
            geometry: THREE.BufferGeometry;
            maxY: number;
            baseHsl: { h: number; s: number; l: number };
        }[] = [];
        const hsl = { h: 0, s: 0, l: 0 };
        for (let i = 0; i < variantCount; i++) {
            const { geometry, maxY } = buildVariant(
                SEED + VARIANT_SEED_OFFSET + i,
            );
            const tintAttr = new THREE.InstancedBufferAttribute(
                new Float32Array(perVariantCapacity * 3),
                3,
            );
            tintAttr.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute("aTint", tintAttr);
            const seedAttr = new THREE.InstancedBufferAttribute(
                new Float32Array(perVariantCapacity),
                1,
            );
            seedAttr.setUsage(THREE.DynamicDrawUsage);
            geometry.setAttribute("aSeed", seedAttr);
            BRAIN_PALETTE[i % BRAIN_PALETTE.length].getHSL(hsl);
            out.push({
                geometry,
                maxY,
                baseHsl: { h: hsl.h, s: hsl.s, l: hsl.l },
            });
        }
        return out;
    }, [variantCount, perVariantCapacity]);

    const meshRefs = useRef<(THREE.InstancedMesh | null)[]>([]);
    if (meshRefs.current.length !== variantCount) {
        meshRefs.current = new Array(variantCount).fill(null);
    }

    const dummy = useMemo(() => new THREE.Object3D(), []);
    const tmpColor = useMemo(() => new THREE.Color(), []);
    const lastOffset = useRef<{ x: number; z: number }>({ x: NaN, z: NaN });
    const perVariantWriteIdx = useMemo(
        () => new Int32Array(variantCount),
        [variantCount],
    );

    useFrame(() => {
        if (
            worldOffset.x === lastOffset.current.x &&
            worldOffset.z === lastOffset.current.z
        ) {
            return;
        }
        lastOffset.current.x = worldOffset.x;
        lastOffset.current.z = worldOffset.z;

        for (let i = 0; i < variantCount; i++) {
            perVariantWriteIdx[i] = 0;
        }

        const offX = worldOffset.x;
        const offZ = worldOffset.z;
        const half = size / 2;
        const cMinX = Math.floor((offX - half) / CELL_SIZE);
        const cMaxX = Math.ceil((offX + half) / CELL_SIZE);
        const cMinZ = Math.floor((offZ - half) / CELL_SIZE);
        const cMaxZ = Math.ceil((offZ + half) / CELL_SIZE);

        for (let cx = cMinX; cx <= cMaxX; cx++) {
            for (let cz = cMinZ; cz <= cMaxZ; cz++) {
                const wx = (cx + cellHash(cx, cz, 0)) * CELL_SIZE;
                const wz = (cz + cellHash(cx, cz, 1)) * CELL_SIZE;
                if (wx < offX - half || wx > offX + half) continue;
                if (wz < offZ - half || wz > offZ + half) continue;

                const { height: gy, stratum } = sampleTerrain(wx, wz);
                if (
                    stratum < PLACEMENT_STRATUM_MIN ||
                    stratum > PLACEMENT_STRATUM_MAX
                ) {
                    continue;
                }

                const variantIdx = Math.floor(
                    cellHash(cx, cz, 10) * variantCount,
                );
                const variant = variants[variantIdx];
                if (!variant) continue;

                const sizeJitter =
                    SIZE_MIN + cellHash(cx, cz, 2) * (SIZE_MAX - SIZE_MIN);
                if (
                    gy + variant.maxY * sizeJitter >
                    waterY - SUBMERGE_BUFFER
                ) {
                    continue;
                }

                const mask = fbm(
                    wx * PATCH_FREQUENCY,
                    wz * PATCH_FREQUENCY,
                    SEED + PATCH_SEED_OFFSET,
                );
                const patchDensity = smoothstep(
                    PATCH_THRESHOLD - PATCH_EDGE_WIDTH,
                    PATCH_THRESHOLD + PATCH_EDGE_WIDTH,
                    mask,
                );
                if (cellHash(cx, cz, 9) > patchDensity) continue;

                const mesh = meshRefs.current[variantIdx];
                if (!mesh) continue;
                const writeIdx = perVariantWriteIdx[variantIdx];
                if (writeIdx >= perVariantCapacity) continue;

                const yaw = cellHash(cx, cz, 4) * Math.PI * 2;
                dummy.position.set(wx - offX, gy, wz - offZ);
                dummy.rotation.set(0, yaw, 0);
                dummy.scale.set(sizeJitter, sizeJitter, sizeJitter);
                dummy.updateMatrix();
                mesh.setMatrixAt(writeIdx, dummy.matrix);

                const tintAttr = mesh.geometry.attributes
                    .aTint as THREE.InstancedBufferAttribute;
                const tintArr = tintAttr.array as Float32Array;
                const h2 =
                    variant.baseHsl.h +
                    (cellHash(cx, cz, 6) - 0.5) * HUE_JITTER;
                const s2 = Math.max(
                    0,
                    Math.min(
                        1,
                        variant.baseHsl.s +
                            (cellHash(cx, cz, 7) - 0.5) * SAT_JITTER,
                    ),
                );
                const l2 = Math.max(
                    0,
                    Math.min(
                        1,
                        variant.baseHsl.l +
                            (cellHash(cx, cz, 8) - 0.5) * LIGHT_JITTER,
                    ),
                );
                tmpColor.setHSL(h2, s2, l2);
                tintArr[writeIdx * 3 + 0] = tmpColor.r;
                tintArr[writeIdx * 3 + 1] = tmpColor.g;
                tintArr[writeIdx * 3 + 2] = tmpColor.b;

                const seedAttr = mesh.geometry.attributes
                    .aSeed as THREE.InstancedBufferAttribute;
                const seedArr = seedAttr.array as Float32Array;
                seedArr[writeIdx] = cellHash(cx, cz, 11);

                perVariantWriteIdx[variantIdx]++;
            }
        }

        for (let i = 0; i < variantCount; i++) {
            const mesh = meshRefs.current[i];
            if (!mesh) continue;
            mesh.count = perVariantWriteIdx[i];
            mesh.instanceMatrix.needsUpdate = true;
            (
                mesh.geometry.attributes
                    .aTint as THREE.InstancedBufferAttribute
            ).needsUpdate = true;
            (
                mesh.geometry.attributes
                    .aSeed as THREE.InstancedBufferAttribute
            ).needsUpdate = true;
        }
    });

    return (
        <group>
            {variants.map((v, i) => (
                <instancedMesh
                    key={i}
                    ref={(m) => {
                        meshRefs.current[i] = m;
                    }}
                    args={[v.geometry, undefined, perVariantCapacity]}
                    frustumCulled={false}
                >
                    <shaderMaterial
                        vertexShader={vertexShader}
                        fragmentShader={fragmentShader}
                        side={THREE.FrontSide}
                    />
                </instancedMesh>
            ))}
        </group>
    );
}
