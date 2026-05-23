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

// How many distinct branching shapes are baked. Each variant gets its
// own InstancedMesh, so VRAM cost scales linearly — keep modest.
const DEFAULT_VARIANT_COUNT = 6;

// World-space placement grid. Coarser than Seagrass: corals are bigger
// individual objects and we want them spaced, not carpeted.
const CELL_SIZE = 1.4;

// Stratum band. Targets the upper shelf and the start of the reef
// slope — the band of terrain that's elevated enough to read as
// "reef-y" but still below the waterline at typical heights. The reef
// peak itself (n > ~0.72) sits above water and gets rejected by the
// height check below anyway, but capping the band here saves the
// sampling effort.
const PLACEMENT_STRATUM_MIN = 0.5;
const PLACEMENT_STRATUM_MAX = 0.72;

// Patch mask drives clumping. Distinct seed offset from Seagrass so
// colonies and meadows aren't co-located on the same noise contours.
const PATCH_FREQUENCY = 0.14;
const PATCH_THRESHOLD = 0.5;
const PATCH_EDGE_WIDTH = 0.12;
const PATCH_SEED_OFFSET = 199;

// Seed offset for the per-variant L-system PRNG. Decoupled from terrain
// and patch noise so coral shapes don't correlate with reef topology.
const VARIANT_SEED_OFFSET = 1000;

// Tip must stay this far below the waterline at the variant's natural
// scaled height.
const SUBMERGE_BUFFER = 0.15;

// Per-instance uniform scale jitter applied on top of variant shape.
const SIZE_MIN = 0.7;
const SIZE_MAX = 1.3;

// Per-variant base colour. Variant i samples palette[i % len], then
// per-instance HSL jitter perturbs around that — so all instances of a
// variant share a recognisable palette region without being identical.
// Saturated and bright so they pop against the water tint instead of
// reading as dead-wood grey.
const CORAL_PALETTE = [
    new THREE.Color("#ff7a99"), // bright coral pink
    new THREE.Color("#f08a5d"), // peach
    new THREE.Color("#b88be8"), // bright lavender
    new THREE.Color("#f7c95e"), // sun gold
    new THREE.Color("#d6577d"), // deep coral
    new THREE.Color("#7fc8a9"), // sea mint
    new THREE.Color("#ffd087"), // soft amber
    new THREE.Color("#a26ec8"), // grape
];

const HUE_JITTER = 0.05;
const SAT_JITTER = 0.15;
const LIGHT_JITTER = 0.15;

// L-system shape parameters. Children-per-branch is multiplicative, so
// raising MAX_DEPTH or the depth=0 child count inflates vertex counts
// fast — tune in tandem with MIN_RADIUS which provides natural cutoff.
const TRUNK_LENGTH = 0.22;
const TRUNK_RADIUS = 0.075;
const SEGMENTS_PER_BRANCH = 3;
const MAX_DEPTH = 3;
const MIN_RADIUS = 0.015;
const LENGTH_DECAY_MIN = 0.5;
const LENGTH_DECAY_MAX = 0.72;
// Pitch of a child branch off its parent's direction, in radians.
// Bigger spread keeps children sideways rather than parallel to parent —
// otherwise branches stack upward and the coral grows tall instead of
// bushy, and tall coral fails the waterline check.
const BRANCH_PITCH_MIN = 0.6;
const BRANCH_PITCH_MAX = 1.3;
// Per-segment random direction nudge + upward bias, so a branch curls
// instead of running perfectly straight. UPBIAS kept small so branches
// don't all eventually point up.
const SEGMENT_CURL = 0.08;
const SEGMENT_UPBIAS = 0.025;
// Cross-section faceting. 6 reads as smooth on a thick branch without
// paying for 8+; we bumped from 5 since branches are now chunkier and
// faceting becomes visible.
const SIDES_PER_TUBE = 6;

// Tip taper. Forking branches narrow strongly so the fork point reads
// as a pinch; terminal branches keep most of their radius for a slight
// bulbous tip (common in real Pocillopora / Acropora).
const TIP_RATIO_FORKING = 0.55;
// Slightly >1: terminal branches widen toward the tip rather than
// narrowing, giving the club-shaped polyp-zone look real Acropora /
// Pocillopora tips have.
const TIP_RATIO_TERMINAL = 1.15;
// Child branch starts at this fraction of the parent's tip radius —
// near 1.0 keeps the fork visually continuous (no width jump).
const CHILD_START_RATIO_MIN = 0.85;
const CHILD_START_RATIO_MAX = 1.0;

// Multi-trunk colony base. Several near-vertical stems emerge from a
// small basal disc, splayed slightly outward — this is what makes the
// silhouette read as "coral colony" rather than "tree".
const BASE_TRUNKS_MIN = 3;
const BASE_TRUNKS_MAX = 5;
const BASE_DISC_RADIUS = TRUNK_RADIUS * 1.4;
const BASE_TILT_MIN = 0.1;
const BASE_TILT_MAX = 0.35;

// Seedable PRNG. Same seed → identical branching shape, so variant
// geometry is stable across reloads.
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

// Same per-cell hash family as Seagrass. Channel index switches the
// sub-stream so placement/jitter/colour use independent randoms.
function cellHash(cx: number, cz: number, channel: number): number {
    const s = Math.sin(cx * 127.1 + cz * 311.7 + channel * 53.7) * 43758.5453;
    return s - Math.floor(s);
}

// Return a unit vector pitched off `parent` by `pitch` radians, with
// azimuth `yaw` around the parent axis. Used to fork children with
// controllable spread + rotation.
function deviateDirection(
    parent: THREE.Vector3,
    pitch: number,
    yaw: number,
): THREE.Vector3 {
    // Picking a stable reference avoids a degenerate cross product when
    // `parent` is itself near-vertical.
    const refUp =
        Math.abs(parent.y) < 0.9
            ? new THREE.Vector3(0, 1, 0)
            : new THREE.Vector3(1, 0, 0);
    const e1 = new THREE.Vector3().crossVectors(parent, refUp).normalize();
    const e2 = new THREE.Vector3().crossVectors(parent, e1).normalize();
    const perp = e1
        .clone()
        .multiplyScalar(Math.cos(yaw))
        .addScaledVector(e2, Math.sin(yaw));
    return parent
        .clone()
        .multiplyScalar(Math.cos(pitch))
        .addScaledVector(perp, Math.sin(pitch))
        .normalize();
}

// Append a tapered cylinder section from `a` (radius ra) to `b`
// (radius rb) into the shared positions/normals/indices arrays. Each
// vertex's normal points radially outward, giving smooth shading
// around the cylinder. `ta`/`tb` are tip-blend values written to the
// per-vertex tValues array — used by the fragment shader to highlight
// the last ~40% of terminal branches.
function addTube(
    positions: number[],
    normals: number[],
    indices: number[],
    tValues: number[],
    a: THREE.Vector3,
    b: THREE.Vector3,
    ra: number,
    rb: number,
    ta: number,
    tb: number,
    sides: number,
): void {
    const dir = new THREE.Vector3().subVectors(b, a).normalize();
    const refUp =
        Math.abs(dir.y) < 0.9
            ? new THREE.Vector3(0, 1, 0)
            : new THREE.Vector3(1, 0, 0);
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
        tValues.push(ta);
        positions.push(b.x + ox * rb, b.y + oy * rb, b.z + oz * rb);
        normals.push(ox, oy, oz);
        tValues.push(tb);
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

// Walk a single branch as a chain of SEGMENTS_PER_BRANCH tubes, curling
// the direction each segment, then fork children at the tip. Termination
// is by depth or by radius falling below MIN_RADIUS — whichever hits
// first, which decouples shape detail from the depth limit.
function growBranch(
    positions: number[],
    normals: number[],
    indices: number[],
    tValues: number[],
    base: THREE.Vector3,
    dir: THREE.Vector3,
    length: number,
    radius: number,
    depth: number,
    rng: () => number,
): void {
    if (depth > MAX_DEPTH || radius < MIN_RADIUS) return;

    // Decide branching up front — childCount drives both the tip taper
    // (forks narrow more than terminals) and the per-child yaw layout.
    let childCount =
        depth < 1 ? 4 : depth < 2 ? 3 : depth < 3 ? (rng() < 0.5 ? 3 : 2) : 0;
    // Predict whether any child would survive the depth+1 entry check.
    // If not, mark this branch terminal — otherwise the tip taper picks
    // the forking ratio (strong narrowing into a pinch) and the visible
    // result is a thin point with no children fanning out of it.
    const forkingTipR = radius * TIP_RATIO_FORKING;
    const minPredictedChildR = forkingTipR * CHILD_START_RATIO_MIN;
    if (
        childCount > 0 &&
        (depth + 1 > MAX_DEPTH || minPredictedChildR < MIN_RADIUS)
    ) {
        childCount = 0;
    }
    const isTerminal = childCount === 0;
    const rTip = radius * (isTerminal ? TIP_RATIO_TERMINAL : TIP_RATIO_FORKING);

    let p = base.clone();
    let d = dir.clone();
    const segLen = length / SEGMENTS_PER_BRANCH;

    for (let i = 0; i < SEGMENTS_PER_BRANCH; i++) {
        const r0 = lerp(radius, rTip, i / SEGMENTS_PER_BRANCH);
        const r1 = lerp(radius, rTip, (i + 1) / SEGMENTS_PER_BRANCH);
        // Tip-blend t is non-zero only on terminal branches: ramps 0→1
        // from this branch's base to its tip. Non-terminals always
        // write 0, suppressing the highlight in the colony interior.
        const t0 = isTerminal ? i / SEGMENTS_PER_BRANCH : 0;
        const t1 = isTerminal ? (i + 1) / SEGMENTS_PER_BRANCH : 0;
        const next = p.clone().addScaledVector(d, segLen);
        addTube(
            positions,
            normals,
            indices,
            tValues,
            p,
            next,
            r0,
            r1,
            t0,
            t1,
            SIDES_PER_TUBE,
        );
        p = next;
        d = d
            .clone()
            .add(
                new THREE.Vector3(
                    (rng() - 0.5) * SEGMENT_CURL,
                    SEGMENT_UPBIAS,
                    (rng() - 0.5) * SEGMENT_CURL,
                ),
            )
            .normalize();
    }

    // Children fan out instead of clumping: distribute base yaws evenly
    // around the parent axis, then jitter within one slice's width.
    const yawJitter = childCount > 0 ? Math.PI / childCount : 0;
    for (let i = 0; i < childCount; i++) {
        const baseYaw = (i / Math.max(1, childCount)) * Math.PI * 2;
        const yaw = baseYaw + (rng() - 0.5) * yawJitter;
        const pitch =
            BRANCH_PITCH_MIN + rng() * (BRANCH_PITCH_MAX - BRANCH_PITCH_MIN);
        const childDir = deviateDirection(d, pitch, yaw);
        const childLen =
            length *
            (LENGTH_DECAY_MIN + rng() * (LENGTH_DECAY_MAX - LENGTH_DECAY_MIN));
        // Child starts at ~parent's tip radius so the fork joint is
        // visually continuous — no diameter jump at the branching point.
        const childRad =
            rTip *
            (CHILD_START_RATIO_MIN +
                rng() * (CHILD_START_RATIO_MAX - CHILD_START_RATIO_MIN));
        growBranch(
            positions,
            normals,
            indices,
            tValues,
            p,
            childDir,
            childLen,
            childRad,
            depth + 1,
            rng,
        );
    }
}

// Bake one variant from a seed. Returns the merged geometry plus the
// natural max-Y (in unit scale) so the placement loop can do a per-
// variant waterline-clearance check.
function buildVariant(seed: number): {
    geometry: THREE.BufferGeometry;
    maxY: number;
} {
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    const tValues: number[] = [];
    const rng = mulberry32(seed);

    // Multi-trunk colony base. Several near-vertical stems emerge from
    // points distributed around a small disc, each tilted outward by a
    // few degrees so they splay rather than running parallel.
    const trunkCount =
        BASE_TRUNKS_MIN +
        Math.floor(rng() * (BASE_TRUNKS_MAX - BASE_TRUNKS_MIN + 1));
    for (let i = 0; i < trunkCount; i++) {
        const yaw = (i / trunkCount) * Math.PI * 2 + (rng() - 0.5) * 0.4;
        const r = BASE_DISC_RADIUS * (0.4 + rng() * 0.6);
        const basePos = new THREE.Vector3(
            Math.cos(yaw) * r,
            0,
            Math.sin(yaw) * r,
        );
        const tilt = BASE_TILT_MIN + rng() * (BASE_TILT_MAX - BASE_TILT_MIN);
        // Tilt direction outward from the colony centre (the same yaw
        // as the basal position) so trunks lean away from each other.
        const trunkDir = new THREE.Vector3(
            Math.cos(yaw) * Math.sin(tilt),
            Math.cos(tilt),
            Math.sin(yaw) * Math.sin(tilt),
        ).normalize();
        const trunkLen = TRUNK_LENGTH * (0.85 + rng() * 0.35);
        growBranch(
            positions,
            normals,
            indices,
            tValues,
            basePos,
            trunkDir,
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
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geo.setAttribute("aBranchT", new THREE.Float32BufferAttribute(tValues, 1));
    geo.setIndex(indices);

    return { geometry: geo, maxY };
}

const vertexShader = /* glsl */ `
    precision highp float;

    attribute vec3 aTint;
    attribute float aBranchT;

    varying vec3 vColor;
    varying vec3 vNormal;
    varying float vBranchT;

    void main() {
        vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
        vec3 instanceNormal = mat3(instanceMatrix) * normal;
        vNormal = normalize(normalMatrix * instanceNormal);
        vColor = aTint;
        vBranchT = aBranchT;
    }
`;

const fragmentShader = /* glsl */ `
    precision highp float;

    varying vec3 vColor;
    varying vec3 vNormal;
    varying float vBranchT;

    // Warm cream tip target — reads as concentrated polyp tissue against
    // the saturated branch colour.
    const vec3 TIP_COLOR = vec3(0.97, 0.93, 0.82);
    // Highlight ramps in over the last portion of a terminal branch.
    const float TIP_RAMP_START = 0.4;
    const float TIP_RAMP_END = 1.0;
    // Peak blend at the very tip; <1 keeps a hint of the variant hue.
    const float TIP_BLEND_PEAK = 0.8;

    void main() {
        vec3 lightDir = normalize(vec3(10.0, 15.0, 8.0));
        float ndotl = abs(dot(normalize(vNormal), lightDir));
        float lit = 0.45 + 0.55 * ndotl;
        float tipMix =
            smoothstep(TIP_RAMP_START, TIP_RAMP_END, vBranchT) * TIP_BLEND_PEAK;
        vec3 base = mix(vColor, TIP_COLOR, tipMix);
        gl_FragColor = vec4(base * lit, 1.0);
    }
`;

export default function Coral({
    size,
    waterY,
    variantCount = DEFAULT_VARIANT_COUNT,
}: Props) {
    // Worst-case visible cells in the slice (same +3 alignment slop as
    // Seagrass).
    const targetCells = useMemo(() => {
        const cellsPerSide = Math.ceil(size / CELL_SIZE) + 3;
        return cellsPerSide * cellsPerSide;
    }, [size]);

    // Per-variant capacity. Hash-uniform bucketing gives ~targetCells /
    // variantCount per bucket on average; ×2 absorbs unevenness at small
    // N and the +16 covers very low N or small slices.
    const perVariantCapacity = useMemo(
        () => Math.ceil((targetCells * 2) / variantCount) + 16,
        [targetCells, variantCount],
    );

    // Bake variants once (or rebuild only if variantCount /
    // perVariantCapacity changes — which won't happen at runtime).
    // Geometry, the unit-scale Y extent, and the variant's base HSL are
    // bundled so the per-frame loop has everything it needs from one
    // lookup.
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
            CORAL_PALETTE[i % CORAL_PALETTE.length].getHSL(hsl);
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
        // Placement is deterministic in world coords, so visible
        // instances only need rewriting when the slice has actually
        // panned.
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

                // Pick variant first — the height check below depends
                // on the variant's natural max-Y.
                const variantIdx = Math.floor(
                    cellHash(cx, cz, 10) * variantCount,
                );
                const variant = variants[variantIdx];
                if (!variant) continue;

                const sizeJitter =
                    SIZE_MIN + cellHash(cx, cz, 2) * (SIZE_MAX - SIZE_MIN);
                if (gy + variant.maxY * sizeJitter > waterY - SUBMERGE_BUFFER) {
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

                perVariantWriteIdx[variantIdx]++;
            }
        }

        for (let i = 0; i < variantCount; i++) {
            const mesh = meshRefs.current[i];
            if (!mesh) continue;
            mesh.count = perVariantWriteIdx[i];
            mesh.instanceMatrix.needsUpdate = true;
            (
                mesh.geometry.attributes.aTint as THREE.InstancedBufferAttribute
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
