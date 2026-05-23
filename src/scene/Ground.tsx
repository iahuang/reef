import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { sampleTerrain } from "./terrain";
import { worldOffset } from "./worldOffset";

type Props = {
    size: number;
    // Local-frame Y of the water surface. Used by the caustic shader to
    // mask emission to the underwater portion of the seabed.
    waterY: number;
    // Local-frame Y of the seabed slab's underside. Sets how tall the
    // visible rim under the tank reads — independent of where the dune
    // heightfield sits.
    baseY: number;
};

// Number of quads per side on the heightfield.
const SUBDIVISIONS = 128;

// How far the terrain extends past the requested `size` on each side.
// Keeps the side-wall planes from coinciding with the water's translucent
// side panels (which sit exactly at ±size/2), which would otherwise
// z-fight. Small enough that the overhang stays hidden behind the panels.
const OVERHANG = 0.01;

// ---- Caustics tuning ----
// Spatial density of the Voronoi cells (1/world units). Larger → smaller,
// finer-grained cells. The Blender reference uses Mapping Scale on a unit
// Voronoi, which is equivalent to multiplying world XZ by this factor.
const CAUSTIC_SCALE = 5.5;

// Rate at which the time axis of the 3D Voronoi advances (world units per
// second). Controls how fast the pattern shimmers; the Blender reference
// animated the Mapping Z value, which lives in the same units.
const CAUSTIC_SPEED = 0.5;

// Overall brightness multiplier on the caustic emission. Stands in for
// the Blender 200× Light Intensity multiplier; tuned down because three's
// emissive accumulator is already in linear-light units, not lumens.
const CAUSTIC_INTENSITY = 8;

// Caustic tint — slightly cool, slightly bluish white. Mixes with the
// warm sand colour so the bright streaks read as refracted sky light
// rather than additive white.
const CAUSTIC_COLOR = new THREE.Color("#d8ecff");

export default function Ground({ size, waterY, baseY }: Props) {
    const geometry = useMemo(() => {
        const N = SUBDIVISIONS;
        const extent = size + 2 * OVERHANG;
        const half = extent / 2;
        const step = extent / N;

        const positions: number[] = [];
        const indices: number[] = [];

        // ---- Top heightfield ----
        // (N+1)×(N+1) grid. Row i fixes z, column j fixes x; y comes
        // from the shared `sampleTerrain` function so foliage placement
        // sees the exact same seabed.
        const topIndex = (i: number, j: number) => i * (N + 1) + j;
        for (let i = 0; i <= N; i++) {
            for (let j = 0; j <= N; j++) {
                const x = -half + j * step;
                const z = -half + i * step;
                const { height } = sampleTerrain(x, z);
                positions.push(x, height, z);
            }
        }
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const a = topIndex(i, j);
                const b = topIndex(i, j + 1);
                const c = topIndex(i + 1, j);
                const d = topIndex(i + 1, j + 1);
                indices.push(a, c, b, b, c, d);
            }
        }

        // ---- Side walls ----
        // For each perimeter edge of the heightfield, allocate N+1 bottom
        // vertices at y = baseY that mirror the top edge's (x, z), then
        // stitch quads between the top and bottom edges. `flip` selects
        // winding so the outward face points away from the tank centre —
        // worked out from the cross product of the triangle's edge
        // vectors for each axis.
        const addSide = (topEdge: number[], flip: boolean) => {
            const bottomStart = positions.length / 3;
            for (const ti of topEdge) {
                const x = positions[ti * 3];
                const z = positions[ti * 3 + 2];
                positions.push(x, baseY, z);
            }
            for (let k = 0; k < topEdge.length - 1; k++) {
                const tA = topEdge[k];
                const tB = topEdge[k + 1];
                const bA = bottomStart + k;
                const bB = bottomStart + k + 1;
                if (!flip) {
                    indices.push(tA, bA, tB, tB, bA, bB);
                } else {
                    indices.push(tA, tB, bA, tB, bB, bA);
                }
            }
        };

        const edgeMinusZ: number[] = [];
        const edgePlusZ: number[] = [];
        const edgeMinusX: number[] = [];
        const edgePlusX: number[] = [];
        for (let j = 0; j <= N; j++) {
            edgeMinusZ.push(topIndex(0, j));
            edgePlusZ.push(topIndex(N, j));
        }
        for (let i = 0; i <= N; i++) {
            edgeMinusX.push(topIndex(i, 0));
            edgePlusX.push(topIndex(i, N));
        }

        addSide(edgePlusZ, false);
        addSide(edgeMinusZ, true);
        addSide(edgeMinusX, false);
        addSide(edgePlusX, true);

        // ---- Bottom face ----
        // Single quad at y = baseY, normal pointing -Y. Seals the mesh
        // into a watertight solid.
        const floorStart = positions.length / 3;
        positions.push(-half, baseY, -half);
        positions.push(half, baseY, -half);
        positions.push(-half, baseY, half);
        positions.push(half, baseY, half);
        indices.push(
            floorStart,
            floorStart + 1,
            floorStart + 2,
            floorStart + 1,
            floorStart + 3,
            floorStart + 2,
        );

        const geo = new THREE.BufferGeometry();
        geo.setAttribute(
            "position",
            new THREE.Float32BufferAttribute(positions, 3),
        );
        geo.setIndex(indices);
        geo.computeVertexNormals();
        return geo;
    }, [size, baseY]);

    // Uniforms held in a ref so the same object instance is reused for the
    // life of the material — onBeforeCompile binds these by reference, and
    // useFrame mutates uTime.value in place each frame.
    const uniforms = useRef({
        uTime: { value: 0 },
        uWaterLevel: { value: waterY },
        uCausticScale: { value: CAUSTIC_SCALE },
        uCausticSpeed: { value: CAUSTIC_SPEED },
        uCausticIntensity: { value: CAUSTIC_INTENSITY },
        uCausticColor: { value: CAUSTIC_COLOR.clone() },
        // World offset applied to caustic sampling so the pattern stays
        // anchored to the underlying noise-world coordinates as the slice
        // pans, rather than scrolling along with the mesh.
        uCausticOffset: { value: new THREE.Vector2(0, 0) },
    });

    useEffect(() => {
        uniforms.current.uWaterLevel.value = waterY;
    }, [waterY]);

    // Custom MeshStandardMaterial: the base PBR shading is unchanged; we
    // only inject a caustic emissive term that lights up the underwater,
    // upward-facing parts of the seabed. Built once and patched via
    // onBeforeCompile.
    const material = useMemo(() => {
        const m = new THREE.MeshStandardMaterial({
            color: "#c2a878",
            roughness: 0.95,
            metalness: 0,
            flatShading: true,
        });

        m.onBeforeCompile = (shader) => {
            shader.uniforms.uTime = uniforms.current.uTime;
            shader.uniforms.uWaterLevel = uniforms.current.uWaterLevel;
            shader.uniforms.uCausticScale = uniforms.current.uCausticScale;
            shader.uniforms.uCausticSpeed = uniforms.current.uCausticSpeed;
            shader.uniforms.uCausticIntensity =
                uniforms.current.uCausticIntensity;
            shader.uniforms.uCausticColor = uniforms.current.uCausticColor;
            shader.uniforms.uCausticOffset = uniforms.current.uCausticOffset;

            // Vertex: export mesh-local position and normal. The water Y
            // and Voronoi sample both live in this frame, so we don't pay
            // the cost of a modelMatrix multiply per fragment.
            shader.vertexShader = shader.vertexShader
                .replace(
                    "#include <common>",
                    `#include <common>
                    varying vec3 vCausticPos;
                    varying vec3 vCausticNormal;`,
                )
                .replace(
                    "#include <begin_vertex>",
                    `#include <begin_vertex>
                    vCausticPos = transformed;
                    vCausticNormal = normalize(objectNormal);`,
                );

            // Fragment: 3D Voronoi F1 distance → c * log_c(0.994) falloff,
            // gated by an upward-facing factor and an underwater step.
            // Faithful port of the Pattern / Light Falloff / Light
            // Intensity groups in the reference Blender graph.
            shader.fragmentShader = shader.fragmentShader
                .replace(
                    "#include <common>",
                    `#include <common>
                    uniform float uTime;
                    uniform float uWaterLevel;
                    uniform float uCausticScale;
                    uniform float uCausticSpeed;
                    uniform float uCausticIntensity;
                    uniform vec3  uCausticColor;
                    uniform vec2  uCausticOffset;
                    varying vec3 vCausticPos;
                    varying vec3 vCausticNormal;

                    // Cheap pseudo-random vec3 in [0,1]^3 keyed on a cell
                    // coordinate. Equivalent in role to Blender's per-cell
                    // hashed offset with Randomness = 1.
                    vec3 caustic_hash33(vec3 p) {
                        p = vec3(
                            dot(p, vec3(127.1, 311.7,  74.7)),
                            dot(p, vec3(269.5, 183.3, 246.1)),
                            dot(p, vec3(113.5, 271.9, 124.6))
                        );
                        return fract(sin(p) * 43758.5453);
                    }

                    // 3D Voronoi F1 (Euclidean) distance: the smallest
                    // distance from p to any feature point in the 3x3x3
                    // cell neighbourhood around floor(p).
                    float caustic_voronoiF1(vec3 p) {
                        vec3 ip = floor(p);
                        vec3 fp = fract(p);
                        float best = 1.0e9;
                        for (int z = -1; z <= 1; z++) {
                            for (int y = -1; y <= 1; y++) {
                                for (int x = -1; x <= 1; x++) {
                                    vec3 offs = vec3(float(x), float(y), float(z));
                                    vec3 feat = offs + caustic_hash33(ip + offs);
                                    best = min(best, length(feat - fp));
                                }
                            }
                        }
                        return best;
                    }`,
                )
                .replace(
                    "#include <emissivemap_fragment>",
                    `#include <emissivemap_fragment>
                    {
                        // Pattern: scale only the horizontal axes, animate
                        // the third axis as a function of time. Adding
                        // uCausticOffset shifts the sample point so the
                        // pattern is keyed to noise-world coordinates and
                        // pans alongside the terrain.
                        vec3 p = vec3(
                            (vCausticPos.xz + uCausticOffset) * uCausticScale,
                            uTime * uCausticSpeed
                        );
                        float c = caustic_voronoiF1(p);

                        // Light Falloff: c * log_c(0.994), with min() to
                        // keep log finite as c approaches 1.
                        float cClamp  = min(c, 0.999);
                        float sharp   = log(0.994) / log(cClamp);
                        float falloff = c * sharp;

                        // Light Intensity: upward-facing factor stands in
                        // for the Normal-Z map range in the reference.
                        float facing = smoothstep(
                            0.0, 1.0,
                            normalize(vCausticNormal).y
                        );

                        // Hard underwater cut — the above-water plateau
                        // sits just above uWaterLevel and must stay dark.
                        float underwater = step(vCausticPos.y, uWaterLevel);

                        totalEmissiveRadiance +=
                            uCausticColor *
                            falloff *
                            facing *
                            underwater *
                            uCausticIntensity;
                    }`,
                );
        };

        return m;
    }, []);

    useEffect(() => () => material.dispose(), [material]);

    // Tracks the offset the heightfield was last rebuilt at, so we can skip
    // the rebuild entirely when the slice is stationary.
    const lastBuiltOffset = useRef({ x: NaN, z: NaN });

    useFrame((_, delta) => {
        uniforms.current.uTime.value += delta;

        if (
            worldOffset.x === lastBuiltOffset.current.x &&
            worldOffset.z === lastBuiltOffset.current.z
        ) {
            return;
        }
        lastBuiltOffset.current.x = worldOffset.x;
        lastBuiltOffset.current.z = worldOffset.z;

        // The heightfield's XZ grid is fixed in local space; only the Y
        // values change as the offset advances. Side-wall bottoms and the
        // floor quad sit past the top-grid vertex range, so they're left
        // alone at baseY. Top-edge vertices on the side walls share indices
        // with the heightfield, so updating the grid covers them too.
        const N = SUBDIVISIONS;
        const extent = size + 2 * OVERHANG;
        const half = extent / 2;
        const step = extent / N;
        const posAttr = geometry.attributes.position as THREE.BufferAttribute;
        const arr = posAttr.array as Float32Array;

        const offX = worldOffset.x;
        const offZ = worldOffset.z;
        for (let i = 0; i <= N; i++) {
            for (let j = 0; j <= N; j++) {
                const idx = (i * (N + 1) + j) * 3;
                const localX = -half + j * step;
                const localZ = -half + i * step;
                const { height } = sampleTerrain(localX + offX, localZ + offZ);
                arr[idx + 1] = height;
            }
        }
        posAttr.needsUpdate = true;
        // The caustic shader uses vCausticNormal (smooth per-vertex
        // normal) for its facing factor; keep it in sync so the falloff
        // tracks the new surface as it scrolls.
        geometry.computeVertexNormals();

        uniforms.current.uCausticOffset.value.set(offX, offZ);
    });

    return <mesh geometry={geometry} material={material} receiveShadow />;
}
