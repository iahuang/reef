import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { SEED, fbm, lerp, sampleTerrain, smoothstep } from "./terrain";
import { worldOffset } from "./worldOffset";

type Props = {
    size: number;
    waterY: number;
    variantCount?: number;
};

const DEFAULT_VARIANT_COUNT = 5;

// Placement — sea fans grow in current-rich areas: shelf and lower reef
// slope. Larger CELL_SIZE than branching coral since real sea fans are
// rare individual specimens, not carpeting colonies.
const CELL_SIZE = 2.5;
const PLACEMENT_STRATUM_MIN = 0.45;
const PLACEMENT_STRATUM_MAX = 0.65;

// Patch mask — distinct seed so fans cluster on different contours than
// seagrass (99), branching coral (199), or brain coral (311).
const PATCH_FREQUENCY = 0.13;
const PATCH_THRESHOLD = 0.55;
const PATCH_EDGE_WIDTH = 0.1;
const PATCH_SEED_OFFSET = 421;

const VARIANT_SEED_OFFSET = 3000;
const SUBMERGE_BUFFER = 0.15;
const SIZE_MIN = 0.7;
const SIZE_MAX = 1.3;

// Current direction in world XZ — duplicated from Seagrass's local
// CURRENT_DIR so fans orient consistently with the same flow that
// drives blade sway. Vector2.y here represents the world Z component
// (standard Three.js 2D-on-XZ convention).
const CURRENT_DIR = new THREE.Vector2(1.0, 0.3).normalize();
// Random yaw jitter on top of the current-aligned base yaw, so fans
// don't all face *exactly* the same way — real colonies have some
// individual orientation variation even in a steady current.
const YAW_JITTER = 0.45;

// Vivid Gorgonia palette — sea fans naturally come in saturated reds,
// oranges, purples, yellows. Brighter than the other coral palettes so
// they pop as feature specimens.
const FAN_PALETTE = [
    new THREE.Color("#d6334a"), // crimson
    new THREE.Color("#e8772d"), // burnt orange
    new THREE.Color("#a747c8"), // royal purple
    new THREE.Color("#e8c93e"), // sun yellow
    new THREE.Color("#d65a8b"), // hot pink
    new THREE.Color("#f0a050"), // amber
];
const HUE_JITTER = 0.04;
const SAT_JITTER = 0.1;
const LIGHT_JITTER = 0.12;

// 2D L-system parameters. Branches grow strictly in the fan's plane
// (XY in object space, z=0); thin tubes give visible thickness without
// breaking the planar read from face-on.
//
// Trunk is short and thin — sea fans don't have a tree-like primary
// stem, they radiate from a basal cluster of nearly-equal-thickness
// branches. RADIUS_DECAY is slow (close to 1.0) for the same reason:
// keeps the whole net at near-uniform thickness instead of producing
// thick trunk / thin twig contrast.
const TRUNK_LENGTH = 0.1;
const TRUNK_RADIUS = 0.02;
const SEGMENTS_PER_BRANCH = 2;
const MAX_DEPTH = 5;
const MIN_RADIUS = 0.008;
const LENGTH_DECAY_MIN = 0.65;
const LENGTH_DECAY_MAX = 0.82;
const RADIUS_DECAY_MIN = 0.78;
const RADIUS_DECAY_MAX = 0.92;
// Fork half-angle in radians: each of two children diverges by this
// from parent direction, in opposite signs. Total fork = 2x this.
const BRANCH_FORK_MIN = 0.35;
const BRANCH_FORK_MAX = 0.65;
// Per-segment random direction nudge (in fan plane only). No upward
// bias — that's what makes branches curl back toward vertical and
// gives the silhouette a tree shape. Lateral spread is instead
// provided by the multi-trunk base.
const SEGMENT_CURL = 0.06;
const SIDES_PER_TUBE = 5;
const TIP_TAPER = 0.85;

// Multi-trunk base — the key structural fix vs. a single trunk. 3–4
// stems radiate from the origin across BASE_SPREAD radians (~85°),
// distributed evenly with small per-trunk angle jitter. This gives the
// fan its wide-base, radial-flare silhouette before any subdivision
// happens.
const BASE_TRUNK_COUNT_MIN = 3;
const BASE_TRUNK_COUNT_MAX = 4;
const BASE_SPREAD = 1.5;
const BASE_ANGLE_JITTER = 0.2;
const BASE_LENGTH_JITTER = 0.4;

// Fan-normal direction used as the consistent refUp for every tube in
// the bake. Because branches are constrained to z=0, this is always
// perpendicular to the branch direction (no degeneracy), and using the
// same refUp throughout avoids cross-section twisting when a branch's
// direction crosses the +Y vs +X majority threshold.
const FAN_NORMAL = new THREE.Vector3(0, 0, 1);

function mulberry32(seed: number): () => number {
    let s = seed | 0;
    return () => {
        s = (s + 0x6d2b79f5) | 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function cellHash(cx: number, cz: number, channel: number): number {
    const s =
        Math.sin(cx * 127.1 + cz * 311.7 + channel * 53.7) * 43758.5453;
    return s - Math.floor(s);
}

// Rotate a vector around the +Z axis (the fan normal), keeping z=0.
// Used to fork branches symmetrically in the fan's plane.
function rotateXY(v: THREE.Vector3, theta: number): THREE.Vector3 {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    return new THREE.Vector3(v.x * c - v.y * s, v.x * s + v.y * c, 0);
}

// Tapered cylinder section between two points. Smooth-shaded (each
// vertex's normal points radially from the segment axis). `refUp` is
// the consistent perpendicular used to build the cross-section frame —
// for fan coral, always the fan normal so tubes don't twist when their
// direction sweeps across the dominant-axis threshold.
function addTube(
    positions: number[],
    normals: number[],
    indices: number[],
    a: THREE.Vector3,
    b: THREE.Vector3,
    ra: number,
    rb: number,
    refUp: THREE.Vector3,
    sides: number,
): void {
    const dir = new THREE.Vector3().subVectors(b, a).normalize();
    const u = new THREE.Vector3().crossVectors(dir, refUp).normalize();
    const v = new THREE.Vector3().crossVectors(dir, u).normalize();

    const start = positions.length / 3;
    for (let s = 0; s < sides; s++) {
        const ang = (s / sides) * Math.PI * 2;
        const cu = Math.cos(ang);
        const cv = Math.sin(ang);
        const ox = u.x * cu + v.x * cv;
        const oy = u.y * cu + v.y * cv;
        const oz = u.z * cu + v.z * cv;
        positions.push(a.x + ox * ra, a.y + oy * ra, a.z + oz * ra);
        normals.push(ox, oy, oz);
        positions.push(b.x + ox * rb, b.y + oy * rb, b.z + oz * rb);
        normals.push(ox, oy, oz);
    }

    for (let s = 0; s < sides; s++) {
        const s2 = (s + 1) % sides;
        const a0 = start + s * 2;
        const a1 = start + s * 2 + 1;
        const b0 = start + s2 * 2;
        const b1 = start + s2 * 2 + 1;
        indices.push(a0, b0, a1, a1, b0, b1);
    }
}

// 2D-constrained branch growth. Each call lays SEGMENTS_PER_BRANCH tube
// sections then forks exactly two children at ±fork radians around the
// parent direction (in the fan plane). Symmetric fork keeps the
// silhouette balanced; asymmetric distortion comes from the per-segment
// curl, not the fork choice.
function growBranch(
    positions: number[],
    normals: number[],
    indices: number[],
    base: THREE.Vector3,
    dir: THREE.Vector3,
    length: number,
    radius: number,
    depth: number,
    rng: () => number,
): void {
    if (depth > MAX_DEPTH || radius < MIN_RADIUS) return;

    let p = base.clone();
    let d = dir.clone();
    d.z = 0;
    if (d.length() < 1e-6) return;
    d.normalize();

    const segLen = length / SEGMENTS_PER_BRANCH;
    const rTip = radius * TIP_TAPER;

    for (let i = 0; i < SEGMENTS_PER_BRANCH; i++) {
        const r0 = lerp(radius, rTip, i / SEGMENTS_PER_BRANCH);
        const r1 = lerp(radius, rTip, (i + 1) / SEGMENTS_PER_BRANCH);
        const next = p.clone().addScaledVector(d, segLen);
        addTube(
            positions,
            normals,
            indices,
            p,
            next,
            r0,
            r1,
            FAN_NORMAL,
            SIDES_PER_TUBE,
        );
        p = next;
        // Curl is in the fan plane only — preserve the planar character.
        // No upward bias: the multi-trunk base provides directional
        // spread; biasing here would curl every branch back toward
        // vertical and re-introduce the tree silhouette.
        d.x += (rng() - 0.5) * SEGMENT_CURL;
        d.y += (rng() - 0.5) * SEGMENT_CURL;
        d.z = 0;
        if (d.length() < 1e-6) d.y = 1;
        d.normalize();
    }

    const fork =
        BRANCH_FORK_MIN +
        rng() * (BRANCH_FORK_MAX - BRANCH_FORK_MIN);
    for (let i = 0; i < 2; i++) {
        const sign = i === 0 ? -1 : 1;
        const childDir = rotateXY(d, sign * fork);
        const childLen =
            length *
            (LENGTH_DECAY_MIN +
                rng() * (LENGTH_DECAY_MAX - LENGTH_DECAY_MIN));
        const childRad =
            radius *
            (RADIUS_DECAY_MIN +
                rng() * (RADIUS_DECAY_MAX - RADIUS_DECAY_MIN));
        growBranch(
            positions,
            normals,
            indices,
            p,
            childDir,
            childLen,
            childRad,
            depth + 1,
            rng,
        );
    }
}

// Bake one fan variant. Multi-trunk base: 3–4 stems emerge from origin
// at angles spanning BASE_SPREAD (~85° fan), each then recurses into
// the planar branching network. This is what gives the silhouette its
// radial flare instead of a tree-like single-trunk shape.
function buildVariant(
    seed: number,
): { geometry: THREE.BufferGeometry; maxY: number } {
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    const rng = mulberry32(seed);

    const trunkCount =
        BASE_TRUNK_COUNT_MIN +
        Math.floor(
            rng() * (BASE_TRUNK_COUNT_MAX - BASE_TRUNK_COUNT_MIN + 1),
        );
    for (let i = 0; i < trunkCount; i++) {
        // Evenly distribute angles across the spread; jitter each so
        // trunks aren't perfectly regular.
        const t =
            trunkCount === 1 ? 0 : i / (trunkCount - 1) - 0.5;
        const angle =
            t * BASE_SPREAD + (rng() - 0.5) * BASE_ANGLE_JITTER;
        const dir = new THREE.Vector3(
            Math.sin(angle),
            Math.cos(angle),
            0,
        );
        const trunkLen =
            TRUNK_LENGTH * (1.0 - BASE_LENGTH_JITTER * 0.5 + rng() * BASE_LENGTH_JITTER);
        growBranch(
            positions,
            normals,
            indices,
            new THREE.Vector3(0, 0, 0),
            dir,
            trunkLen,
            TRUNK_RADIUS,
            0,
            rng,
        );
    }

    let maxY = 0;
    for (let i = 1; i < positions.length; i += 3) {
        if (positions[i] > maxY) maxY = positions[i];
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions, 3),
    );
    geo.setAttribute(
        "normal",
        new THREE.Float32BufferAttribute(normals, 3),
    );
    geo.setIndex(indices);

    return { geometry: geo, maxY };
}

const vertexShader = /* glsl */ `
    precision highp float;

    attribute vec3 aTint;

    varying vec3 vColor;
    varying vec3 vNormal;

    void main() {
        vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
        vec3 instanceNormal = mat3(instanceMatrix) * normal;
        vNormal = normalize(normalMatrix * instanceNormal);
        vColor = aTint;
    }
`;

const fragmentShader = /* glsl */ `
    precision highp float;

    varying vec3 vColor;
    varying vec3 vNormal;

    void main() {
        // Higher ambient floor than the other corals — fans are viewed
        // mostly face-on, so directional shading on the thin tubes adds
        // little contrast and we want the saturated palette to read
        // strongly rather than going muddy in shadow.
        vec3 lightDir = normalize(vec3(10.0, 15.0, 8.0));
        float ndotl = abs(dot(normalize(vNormal), lightDir));
        float lit = 0.7 + 0.3 * ndotl;
        gl_FragColor = vec4(vColor * lit, 1.0);
    }
`;

export default function FanCoral({
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

    // Yaw that puts the fan's local +Z normal along world CURRENT_DIR.
    // After yaw rotation around world Y by `yaw`, local Z = (0,0,1)
    // becomes world (sin yaw, 0, cos yaw); equate to (CURRENT_DIR.x, 0,
    // CURRENT_DIR.y) gives yaw = atan2(CURRENT_DIR.x, CURRENT_DIR.y).
    const baseYaw = useMemo(
        () => Math.atan2(CURRENT_DIR.x, CURRENT_DIR.y),
        [],
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
            FAN_PALETTE[i % FAN_PALETTE.length].getHSL(hsl);
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

                // Current-aligned yaw with per-instance jitter. No
                // π-flip — real sea fans all face into the same flow.
                const yaw =
                    baseYaw + (cellHash(cx, cz, 4) - 0.5) * YAW_JITTER;
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
                        side={THREE.DoubleSide}
                    />
                </instancedMesh>
            ))}
        </group>
    );
}
