import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { SEED, fbm, sampleTerrain, smoothstep } from "./terrain";
import { worldOffset } from "./worldOffset";

type Props = {
  size: number;
  waterY: number;
};

// Blade height range (world units). Sampled per-instance for variety.
const BLADE_HEIGHT_MIN = 0.35;
const BLADE_HEIGHT_MAX = 0.5;

// Blade base / tip widths (world units). Linear taper between them.
const BLADE_BASE_WIDTH = 0.045;
const BLADE_TIP_WIDTH = 0.01;

// Segments along the blade. More = smoother curve when bent. Six is
// enough for a quadratic-looking droop; higher costs verts without much
// visible payoff at this camera distance.
const BLADE_SEGMENTS = 4;

// Stratum value (raw warped fBM, 0..1) inside which seagrass is allowed.
// MIN above the basin so we don't carpet deep water; MAX below the reef
// crest so blades don't appear on the rising rocks. Tuned against the
// thresholds in terrain.ts.
const PLACEMENT_STRATUM_MIN = 0.34;
const PLACEMENT_STRATUM_MAX = 0.66;

// Tip must stay this far below the waterline at rest — sway can briefly
// push it closer, but the surface shouldn't slice through average tips.
const SUBMERGE_BUFFER = 0.2;

// Patch-mask noise. Lower frequency → broader contiguous features →
// larger seagrass meadows separated by stretches of bare seabed.
// Threshold sets the mid-point of acceptance; raise it for sparser
// overall coverage. EDGE_WIDTH straddles the threshold to make a soft
// probabilistic band: cells inside the band accept with probability
// proportional to how deep into the patch they sit, so clumps thin
// out at their edges instead of cutting off along a hard noise contour.
const PATCH_FREQUENCY = 0.18;
const PATCH_THRESHOLD = 0.55;
const PATCH_EDGE_WIDTH = 0.1;

// Sway parameters. AMPLITUDE is the horizontal tip displacement in
// world units at peak; SPATIAL_FREQ varies the phase by world position
// so neighboring blades don't sway in unison.
const SWAY_AMPLITUDE = 0.07;
const SWAY_FREQUENCY = 1.1;
const SPATIAL_FREQ = 0.55;

// Per-instance scale jitter (multiplicative, ±this fraction).
const SCALE_JITTER = 0.2;

// World-space spacing of the per-cell jittered grid. Each cell yields at
// most one candidate blade with a deterministic position and parameters
// keyed to its (cx, cz) integer coords — so any given world coordinate
// always hosts the same blade (or none), and panning the slice just
// uncovers / covers the same fixed field rather than redrawing fresh
// random sets. Sets the density (candidate blades per square world unit).
const CELL_SIZE = 0.22;

// Deterministic per-cell pseudo-random in [0, 1). `channel` switches
// sub-streams so different per-blade randoms (jitter, yaw, tint, …) come
// from independent hashes of the same cell coord.
function cellHash(cx: number, cz: number, channel: number): number {
    const s =
        Math.sin(cx * 127.1 + cz * 311.7 + channel * 53.7) * 43758.5453;
    return s - Math.floor(s);
}

// Base seagrass color. Per-instance HSL jitter is applied around it.
const BASE_COLOR = new THREE.Color("#3aa757");
const HUE_JITTER = 0.04;
const SAT_JITTER = 0.18;
const LIGHT_JITTER = 0.18;

// World direction the current flows in. Bend applies along this in
// world XZ, so all blades lean the same way regardless of their yaw.
const CURRENT_DIR = new THREE.Vector2(1.0, 0.3).normalize();

const vertexShader = /* glsl */ `
    precision highp float;

    uniform float uTime;
    uniform vec2 uCurrentDir;
    uniform float uSwayAmplitude;
    uniform float uSwayFrequency;
    uniform float uSpatialFreq;

    attribute float aPhase;
    attribute vec3 aTint;

    varying vec3 vColor;
    varying vec3 vNormal;
    varying float vBladeY;

    void main() {
        // The blade is built as a unit-height mesh (object y in [0, 1])
        // and scaled to world height by the instance matrix.
        vec3 pos = position;

        // Take the vertex into world space first, then add the bend
        // there. Bend lives in world XZ so all blades respond to the
        // same current direction regardless of their yaw.
        vec4 worldPos = modelMatrix * instanceMatrix * vec4(pos, 1.0);

        // Most bend at the tip, zero at the base. y is object-space so
        // base = 0, tip = 1 regardless of instance height scale.
        float bend = pos.y * pos.y;
        float spatial = dot(worldPos.xz, vec2(uSpatialFreq));
        float sway = sin(uTime * uSwayFrequency + spatial + aPhase);
        worldPos.xz += uCurrentDir * (bend * sway * uSwayAmplitude);

        gl_Position = projectionMatrix * viewMatrix * worldPos;

        // Rotate the object-space normal by the instance yaw so simple
        // lighting tracks the blade's actual facing.
        vec3 instanceNormal = mat3(instanceMatrix) * normal;
        vNormal = normalize(normalMatrix * instanceNormal);
        vColor = aTint;
        vBladeY = pos.y;
    }
`;

const fragmentShader = /* glsl */ `
    precision highp float;

    varying vec3 vColor;
    varying vec3 vNormal;
    varying float vBladeY;

    void main() {
        // Cheap lambert with a fixed sun direction matching the scene's
        // primary directional light. DoubleSide is on so we abs() the
        // dot to keep the back faces lit instead of reading as silhouettes.
        vec3 lightDir = normalize(vec3(10.0, 15.0, 8.0));
        float ndotl = abs(dot(normalize(vNormal), lightDir));
        float lit = 0.45 + 0.55 * ndotl;
        // Slight base-to-tip gradient so blades read as 3D objects, not
        // flat shapes — base a touch darker as if shaded by neighbors.
        float bias = mix(0.78, 1.05, vBladeY);
        gl_FragColor = vec4(vColor * lit * bias, 1.0);
    }
`;

export default function Seagrass({ size, waterY }: Props) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  // Upper bound on simultaneously-visible blades. The per-frame loop
  // visits every grid cell overlapping the slice and emits at most one
  // candidate per cell, so the buffer just needs to fit that worst
  // case. +3 absorbs the fractional alignment of the offset against the
  // cell grid (cMinX..cMaxX can grow by one on each side depending on
  // where offX lands).
  const targetCount = useMemo(() => {
    const cellsPerSide = Math.ceil(size / CELL_SIZE) + 3;
    return cellsPerSide * cellsPerSide;
  }, [size]);

  // Build one unit-height blade geometry: two perpendicular tapered
  // quads forming an X-cross, so the blade has presence from any view
  // angle without needing billboarding. Object-space y ∈ [0, 1].
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];

    const buildQuad = (rotY: number) => {
      const c = Math.cos(rotY);
      const s = Math.sin(rotY);
      const start = positions.length / 3;
      for (let i = 0; i <= BLADE_SEGMENTS; i++) {
        const t = i / BLADE_SEGMENTS;
        const halfW =
          (BLADE_BASE_WIDTH + (BLADE_TIP_WIDTH - BLADE_BASE_WIDTH) * t) * 0.5;
        // Left and right edge vertices at this segment.
        positions.push(-halfW * c, t, -halfW * s);
        positions.push(halfW * c, t, halfW * s);
        // Face normal of the quad (perpendicular to its plane).
        normals.push(-s, 0, c);
        normals.push(-s, 0, c);
      }
      for (let i = 0; i < BLADE_SEGMENTS; i++) {
        const a = start + i * 2;
        const b = a + 1;
        const cc = a + 2;
        const d = a + 3;
        indices.push(a, b, cc, b, d, cc);
      }
    };

    buildQuad(0);
    buildQuad(Math.PI / 2);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    geo.setIndex(indices);

    // Per-instance attribute buffers, pre-allocated at targetCount so
    // the per-frame loop can write into them in place. DynamicDrawUsage
    // hints to the driver that uploads happen often.
    const phaseAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(targetCount),
      1,
    );
    phaseAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aPhase", phaseAttr);
    const tintAttr = new THREE.InstancedBufferAttribute(
      new Float32Array(targetCount * 3),
      3,
    );
    tintAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aTint", tintAttr);

    return geo;
  }, [targetCount]);

  // Reusable scratch objects so the per-frame loop allocates nothing.
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);
  const baseHsl = useMemo(() => {
    const o = { h: 0, s: 0, l: 0 };
    BASE_COLOR.getHSL(o);
    return o;
  }, []);
  const lastOffset = useRef<{ x: number; z: number }>({ x: NaN, z: NaN });

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uCurrentDir: { value: CURRENT_DIR.clone() },
      uSwayAmplitude: { value: SWAY_AMPLITUDE },
      uSwayFrequency: { value: SWAY_FREQUENCY },
      uSpatialFreq: { value: SPATIAL_FREQ },
    }),
    [],
  );

  useFrame((_, delta) => {
    uniforms.uTime.value += delta;

    const mesh = meshRef.current;
    if (!mesh) return;

    // The per-cell layout is deterministic by world coords, so visible
    // blades only need rewriting when the offset has actually moved.
    if (
      worldOffset.x === lastOffset.current.x &&
      worldOffset.z === lastOffset.current.z
    ) {
      return;
    }
    lastOffset.current.x = worldOffset.x;
    lastOffset.current.z = worldOffset.z;

    const phaseAttr = mesh.geometry.attributes
      .aPhase as THREE.InstancedBufferAttribute;
    const tintAttr = mesh.geometry.attributes
      .aTint as THREE.InstancedBufferAttribute;
    const phaseArr = phaseAttr.array as Float32Array;
    const tintArr = tintAttr.array as Float32Array;

    const offX = worldOffset.x;
    const offZ = worldOffset.z;
    const half = size / 2;
    const cMinX = Math.floor((offX - half) / CELL_SIZE);
    const cMaxX = Math.ceil((offX + half) / CELL_SIZE);
    const cMinZ = Math.floor((offZ - half) / CELL_SIZE);
    const cMaxZ = Math.ceil((offZ + half) / CELL_SIZE);

    let writeIdx = 0;
    for (let cx = cMinX; cx <= cMaxX && writeIdx < targetCount; cx++) {
      for (let cz = cMinZ; cz <= cMaxZ && writeIdx < targetCount; cz++) {
        // Jittered world position within this cell.
        const wx = (cx + cellHash(cx, cz, 0)) * CELL_SIZE;
        const wz = (cz + cellHash(cx, cz, 1)) * CELL_SIZE;

        // Jitter can push a candidate into a neighbouring cell, so
        // re-check against the slice rect rather than the cell bounds.
        if (wx < offX - half || wx > offX + half) continue;
        if (wz < offZ - half || wz > offZ + half) continue;

        const { height: gy, stratum } = sampleTerrain(wx, wz);
        if (
          stratum < PLACEMENT_STRATUM_MIN ||
          stratum > PLACEMENT_STRATUM_MAX
        ) {
          continue;
        }

        const heightJitter =
          BLADE_HEIGHT_MIN +
          cellHash(cx, cz, 2) * (BLADE_HEIGHT_MAX - BLADE_HEIGHT_MIN);
        const scaleJitter =
          1 + (cellHash(cx, cz, 3) - 0.5) * 2 * SCALE_JITTER;
        const worldHeight = heightJitter * scaleJitter;
        if (gy + worldHeight > waterY - SUBMERGE_BUFFER) continue;

        const mask = fbm(
          wx * PATCH_FREQUENCY,
          wz * PATCH_FREQUENCY,
          SEED + 99,
        );
        // Soft acceptance band straddling PATCH_THRESHOLD. Cells deep
        // inside a patch always pass; cells well outside always fail;
        // cells in the transition zone accept with probability
        // patchDensity, drawn from a deterministic per-cell hash so the
        // same world cell makes the same coin-flip every visit.
        const patchDensity = smoothstep(
          PATCH_THRESHOLD - PATCH_EDGE_WIDTH,
          PATCH_THRESHOLD + PATCH_EDGE_WIDTH,
          mask,
        );
        if (cellHash(cx, cz, 9) > patchDensity) continue;

        const yaw = cellHash(cx, cz, 4) * Math.PI * 2;
        dummy.position.set(wx - offX, gy, wz - offZ);
        dummy.rotation.set(0, yaw, 0);
        dummy.scale.set(scaleJitter, worldHeight, scaleJitter);
        dummy.updateMatrix();
        mesh.setMatrixAt(writeIdx, dummy.matrix);

        phaseArr[writeIdx] = cellHash(cx, cz, 5) * Math.PI * 2;

        const h2 =
          baseHsl.h + (cellHash(cx, cz, 6) - 0.5) * HUE_JITTER;
        const s2 = Math.max(
          0,
          Math.min(1, baseHsl.s + (cellHash(cx, cz, 7) - 0.5) * SAT_JITTER),
        );
        const l2 = Math.max(
          0,
          Math.min(1, baseHsl.l + (cellHash(cx, cz, 8) - 0.5) * LIGHT_JITTER),
        );
        tmpColor.setHSL(h2, s2, l2);
        tintArr[writeIdx * 3 + 0] = tmpColor.r;
        tintArr[writeIdx * 3 + 1] = tmpColor.g;
        tintArr[writeIdx * 3 + 2] = tmpColor.b;

        writeIdx++;
      }
    }

    mesh.count = writeIdx;
    mesh.instanceMatrix.needsUpdate = true;
    phaseAttr.needsUpdate = true;
    tintAttr.needsUpdate = true;
  });

  // args sets the initial count; the per-frame loop writes mesh.count
  // down to the actual number of accepted blades each refresh.
  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, undefined, targetCount]}
      frustumCulled={false}
    >
      <shaderMaterial
        uniforms={uniforms}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        side={THREE.DoubleSide}
      />
    </instancedMesh>
  );
}
