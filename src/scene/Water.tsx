import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import {
    waterPanelFragmentShader,
    waterPanelVertexShader,
    waterSurfaceFragmentShader,
    waterSurfaceVertexShader,
} from "./waterShader";
import { worldOffset } from "./worldOffset";

type Props = {
    size: number;
    // Local-frame Y of the water surface plane.
    surfaceY: number;
    // Local-frame Y of the bottom edge of the water side panels — set to
    // the terrain underside so the tinted volume reads continuously down
    // to the visible bottom rim of the tank.
    baseY: number;
};

// World-space sun direction. Matches the primary directional light in
// Scene; baked into the water shader as a fixed highlight direction so
// the specular pattern stays stable as the camera rotates.
const SUN_DIRECTION = new THREE.Vector3(10, 15, 8).normalize();

// Per-channel extinction coefficient (1/world units). Red dies fastest so
// the water tints toward teal as the path length grows. Tune all three
// together if you want a different chromaticity, or individually for the
// classic "loses red first" depth shift.
const EXTINCTION = new THREE.Vector3(0.55, 0.25, 0.18);

// Tint of the water itself — substitutes for in-scattered light in the
// volumetric model. Used as the fragment colour by both materials, so the
// surface and the side panels stay chromatically consistent.
const WATER_TINT = new THREE.Color("#42BEBE");

// Hard cap on the sampled path length (world units). When the depth pre-
// pass has nothing behind the water at a given pixel (sampled depth ==
// far plane), the path would otherwise read as infinite and force the
// water to full opacity at a hard silhouette. Cap to roughly the longest
// diagonal across the tank so those areas read as deep tint rather than
// a tear.
const MAX_PATH_LENGTH = 20.0;

// Lower bound on the volumetric alpha. The Beer–Lambert term falls to 0
// where path length → 0 — e.g. at the waterline where terrain breaks the
// surface — which makes the water invisible there and erases the shore.
// A small floor keeps the surface readable as a tinted sheet regardless
// of how shallow the water column is.
const MIN_ALPHA = 0.18;

// Colour the water shifts toward at grazing angles (view direction nearly
// parallel to the surface). Cheap Fresnel proxy — picks a brighter,
// slightly desaturated teal so the front-facing surface reads as the
// deeper water tint while side panels and edge-on geometry read lighter.
// The practical payoff is that the seam between adjacent panels (and
// between a panel and the top sheet) becomes visible despite both being
// "water".
const FRESNEL_TINT = new THREE.Color("#b6e3ec");

// 0 → no Fresnel shift, 1 → full mix toward FRESNEL_TINT at exact grazing.
const FRESNEL_STRENGTH = 0.45;

// Reusable Vector2 to avoid per-frame allocation in the render loop.
const _tmpVec2 = new THREE.Vector2();

export default function Water({ size, surfaceY, baseY }: Props) {
    const groupRef = useRef<THREE.Group>(null);

    // Off-screen target whose depth attachment is exposed as a sampleable
    // texture. The colour attachment is unused but still allocated — three
    // requires a colour target to render to.
    const target = useMemo(() => {
        const t = new THREE.WebGLRenderTarget(1, 1, {
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            depthBuffer: true,
        });
        t.depthTexture = new THREE.DepthTexture(1, 1);
        t.depthTexture.minFilter = THREE.NearestFilter;
        t.depthTexture.magFilter = THREE.NearestFilter;
        return t;
    }, []);

    const { size: viewSize, viewport } = useThree();

    useEffect(() => {
        const w = Math.max(1, Math.floor(viewSize.width * viewport.dpr));
        const h = Math.max(1, Math.floor(viewSize.height * viewport.dpr));
        target.setSize(w, h);
    }, [target, viewSize.width, viewSize.height, viewport.dpr]);

    useEffect(
        () => () => {
            target.depthTexture?.dispose();
            target.dispose();
        },
        [target],
    );

    // Uniforms shared by reference between the surface and panel materials
    // so the render loop only updates them in one place. The surface mesh
    // extends this object with its own surface-specific uniforms.
    const sharedUniforms = useMemo(
        () => ({
            uTime: { value: 0 },
            uSceneDepth: { value: target.depthTexture },
            uResolution: { value: new THREE.Vector2() },
            uCameraNear: { value: 0 },
            uCameraFar: { value: 0 },
            uExtinction: { value: EXTINCTION.clone() },
            uWaterTint: { value: WATER_TINT.clone() },
            uMaxPathLength: { value: MAX_PATH_LENGTH },
            uMinAlpha: { value: MIN_ALPHA },
            uFresnelTint: { value: FRESNEL_TINT.clone() },
            uFresnelStrength: { value: FRESNEL_STRENGTH },
        }),
        [target.depthTexture],
    );

    const surfaceUniforms = useMemo(
        () => ({
            ...sharedUniforms,
            uFoamColor: { value: new THREE.Color("#f4fbff") },
            uHighlightColor: { value: new THREE.Color("#bfe9f2") },
            uSunDirection: { value: SUN_DIRECTION.clone() },
            uWaveAmplitude: { value: 0.15 },
            uWaveFrequency: { value: 1.4 },
            uSpeed: { value: 0.35 },
            uWaveEdgeFalloff: { value: 0.04 },
            uBumpFrequency: { value: 72 },
            uBumpStrength: { value: 0.39 },
            uBumpDrift: { value: new THREE.Vector2(0.2, -0.2) },
            uBumpEvolutionRate: { value: 0.2 },
            uBumpOctaveScales: { value: new THREE.Vector2(1.0, 0.8) },
            uBumpOctaveWeights: { value: new THREE.Vector2(0.6, 0.4) },
            uGlowThreshold: { value: 0.9 },
            uBrightThreshold: { value: 0.967 },
            // Thin bright rim along the waterline (where path length → 0).
            // Width is in world units of path length, not pixels — at the
            // current camera angle/zoom this comes out to a few pixels wide.
            uShoreBandWidth: { value: 0.1 },
            uShoreStrength: { value: 0.6 },
            // World offset applied to wave + bump sampling so both
            // patterns are anchored in noise-world coords and slide
            // through the slice as the offset pans, instead of staying
            // glued to the visible rectangle.
            uWaterOffset: { value: new THREE.Vector2(0, 0) },
            uPlaneSize: { value: size },
        }),
        [sharedUniforms, size],
    );

    // Manual two-pass render. Passing a priority > 0 to useFrame disables
    // R3F's default render so we can interleave a depth pre-pass with the
    // main pass. Pre-pass: hide the water group, render the rest of the
    // scene into `target` for its depth buffer. Main pass: restore the
    // group and render to screen — the water shaders sample the depth
    // texture to compute the volumetric path length per pixel.
    useFrame(({ gl, scene, camera }, delta) => {
        sharedUniforms.uTime.value += delta;

        const persp = camera as THREE.PerspectiveCamera;
        sharedUniforms.uCameraNear.value = persp.near;
        sharedUniforms.uCameraFar.value = persp.far;
        gl.getDrawingBufferSize(_tmpVec2);
        sharedUniforms.uResolution.value.copy(_tmpVec2);
        surfaceUniforms.uWaterOffset.value.set(worldOffset.x, worldOffset.z);

        if (groupRef.current) groupRef.current.visible = false;
        gl.setRenderTarget(target);
        gl.render(scene, camera);
        if (groupRef.current) groupRef.current.visible = true;

        gl.setRenderTarget(null);
        gl.render(scene, camera);
    }, 1);

    const halfSize = size / 2;
    // Vertical extent and centre Y of the four side panels, derived from
    // the surface and base baselines so the tinted volume spans the full
    // gap between them.
    const panelHeight = surfaceY - baseY;
    const sideY = (surfaceY + baseY) / 2;

    return (
        <group ref={groupRef}>
            {/* Water surface */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, surfaceY, 0]}>
                <planeGeometry args={[size, size, 64, 64]} />
                <shaderMaterial
                    uniforms={surfaceUniforms}
                    vertexShader={waterSurfaceVertexShader}
                    fragmentShader={waterSurfaceFragmentShader}
                    transparent={true}
                    depthWrite={false}
                />
            </mesh>

            {/* Four perimeter side panels. DoubleSide so the back-facing
                panels also contribute when the camera flies around, and
                depthWrite=false so the panels don't occlude each other or
                the seabed — the camera reads through them as a tinted
                volume whose opacity tracks the depth-pre-pass path length. */}
            <mesh position={[0, sideY, -halfSize]}>
                <planeGeometry args={[size, panelHeight]} />
                <shaderMaterial
                    uniforms={sharedUniforms}
                    vertexShader={waterPanelVertexShader}
                    fragmentShader={waterPanelFragmentShader}
                    transparent={true}
                    depthWrite={false}
                    side={THREE.DoubleSide}
                />
            </mesh>
            <mesh position={[0, sideY, halfSize]}>
                <planeGeometry args={[size, panelHeight]} />
                <shaderMaterial
                    uniforms={sharedUniforms}
                    vertexShader={waterPanelVertexShader}
                    fragmentShader={waterPanelFragmentShader}
                    transparent={true}
                    depthWrite={false}
                    side={THREE.DoubleSide}
                />
            </mesh>
            <mesh
                position={[halfSize, sideY, 0]}
                rotation={[0, Math.PI / 2, 0]}
            >
                <planeGeometry args={[size, panelHeight]} />
                <shaderMaterial
                    uniforms={sharedUniforms}
                    vertexShader={waterPanelVertexShader}
                    fragmentShader={waterPanelFragmentShader}
                    transparent={true}
                    depthWrite={false}
                    side={THREE.DoubleSide}
                />
            </mesh>
            <mesh
                position={[-halfSize, sideY, 0]}
                rotation={[0, Math.PI / 2, 0]}
            >
                <planeGeometry args={[size, panelHeight]} />
                <shaderMaterial
                    uniforms={sharedUniforms}
                    vertexShader={waterPanelVertexShader}
                    fragmentShader={waterPanelFragmentShader}
                    transparent={true}
                    depthWrite={false}
                    side={THREE.DoubleSide}
                />
            </mesh>
        </group>
    );
}
