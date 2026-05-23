import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

type Props = {
    tankSize: number;
    y: number;
    count?: number;
    maxDashWidthPx?: number;
    dashHeightPx?: number;
    flickerRate?: number;
};

// World-anchored ripples rendered as screen-aligned billboards whose
// footprint is fixed in framebuffer pixels. Each instance has a world XZ
// position; the vertex shader projects it to clip space, snaps the centre
// to the framebuffer pixel grid, and emits a quad sized in whole framebuffer
// pixels around it. The width is driven by a per-instance lifetime envelope
// and discretised to odd-pixel widths (1, 3, 5, ..., maxDashWidthPx) so each
// ripple visibly grows from a 1px speck up to its peak and back, ticking
// through whole-pixel steps. After the nearest-neighbour upscale that reads
// as a crisp stroke that pulses in size, with its position bound to a world
// coordinate — so as the camera orbits the ripples track the water surface
// beneath them instead of staying glued to the screen.
export default function Ripples({
    tankSize,
    y,
    count = 300,
    maxDashWidthPx = 10,
    dashHeightPx = 1,
    flickerRate = 0.45,
}: Props) {
    const matRef = useRef<THREE.ShaderMaterial>(null);

    const geometry = useMemo(() => {
        const g = new THREE.InstancedBufferGeometry();

        // Unit quad in [-1, 1]; the vertex shader scales this to the desired
        // pixel footprint and centres it on each instance's projected anchor.
        const corners = new Float32Array([
            -1, -1, 0, 1, -1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 1, 0,
        ]);
        g.setAttribute("position", new THREE.BufferAttribute(corners, 3));

        const positions = new Float32Array(count * 3);
        const seeds = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            positions[i * 3 + 0] = (Math.random() - 0.5) * tankSize;
            positions[i * 3 + 1] = 0;
            positions[i * 3 + 2] = (Math.random() - 0.5) * tankSize;
            seeds[i] = Math.random();
        }
        g.setAttribute(
            "aInstancePos",
            new THREE.InstancedBufferAttribute(positions, 3),
        );
        g.setAttribute("aSeed", new THREE.InstancedBufferAttribute(seeds, 1));
        g.instanceCount = count;
        // The vertex shader expands the quad in screen space, so a tight
        // bounding sphere on the source geometry isn't meaningful — disable
        // culling on the mesh instead and give a generous sphere here.
        g.boundingSphere = new THREE.Sphere(
            new THREE.Vector3(0, 0, 0),
            tankSize,
        );

        return g;
    }, [tankSize, count]);

    const uniforms = useMemo(
        () => ({
            uTime: { value: 0 },
            uResolution: { value: new THREE.Vector2(1, 1) },
            uDashColor: { value: new THREE.Color("#ffffff") },
            uMaxHalfWidth: { value: maxDashWidthPx / 2 },
            uHalfHeight: { value: dashHeightPx / 2 },
            uFlickerRate: { value: flickerRate },
        }),
        [maxDashWidthPx, dashHeightPx, flickerRate],
    );

    const tmpVec = useMemo(() => new THREE.Vector2(), []);

    useFrame((state, delta) => {
        const mat = matRef.current;
        if (!mat) return;
        mat.uniforms.uTime.value += delta;
        state.gl.getDrawingBufferSize(tmpVec);
        mat.uniforms.uResolution.value.copy(tmpVec);
    });

    return (
        <mesh position={[0, y, 0]} renderOrder={2} frustumCulled={false}>
            <primitive object={geometry} attach="geometry" />
            <shaderMaterial
                ref={matRef}
                uniforms={uniforms}
                vertexShader={vertexShader}
                fragmentShader={fragmentShader}
                transparent
                depthWrite={false}
            />
        </mesh>
    );
}

const vertexShader = /* glsl */ `
    uniform vec2 uResolution;
    uniform float uMaxHalfWidth;
    uniform float uHalfHeight;
    uniform float uTime;
    uniform float uFlickerRate;

    attribute vec3 aInstancePos;
    attribute float aSeed;

    void main() {
        // Per-instance lifetime envelope. A triangle wave keeps each
        // discrete width step on screen for the same duration, so the dash
        // ticks evenly: 0 → 1 → 3 → ... → maxDashWidthPx → ... → 3 → 1 → 0.
        float phase = fract(uTime * uFlickerRate + aSeed * 7.0);
        float envelope = 1.0 - abs(1.0 - 2.0 * phase);

        // Snap envelope to one of (steps + 1) discrete levels — including
        // a zero level so the ripple briefly vanishes at the ends of its
        // cycle instead of holding at width 1.
        float steps = floor(uMaxHalfWidth) + 1.0;
        float n = floor(envelope * (steps + 1.0));
        // n=0 → halfWidth=0 (collapsed); n=k → halfWidth = k - 0.5, giving
        // an on-screen width of (2k - 1) framebuffer pixels (1, 3, 5, ...).
        float halfWidth = max(n - 0.5, 0.0);
        // For n=0 we also drop half-height to zero so the quad collapses
        // fully and produces no fragments.
        float halfHeight = n < 0.5 ? 0.0 : uHalfHeight;

        // Project the world anchor to clip space.
        vec4 centerClip = projectionMatrix * viewMatrix * modelMatrix * vec4(aInstancePos, 1.0);

        // Size of one framebuffer pixel in NDC.
        vec2 pixelGrid = 2.0 / uResolution;

        // Snap the projected anchor to the framebuffer pixel grid so the
        // dash always lands on whole pixels and reads as a crisp block
        // after the nearest-neighbour upscale.
        vec2 centerNdc = floor(centerClip.xy / centerClip.w / pixelGrid + 0.5) * pixelGrid;

        // Offset this vertex by its quad-corner amount, scaled to the
        // per-instance lifetime-driven footprint.
        vec2 cornerNdc = centerNdc + position.xy * vec2(halfWidth, halfHeight) * pixelGrid;

        // Multiply back by w to undo the implicit perspective divide.
        gl_Position = vec4(cornerNdc * centerClip.w, centerClip.z, centerClip.w);
    }
`;

const fragmentShader = /* glsl */ `
    precision highp float;

    uniform vec3 uDashColor;

    void main() {
        // Width carries the lifetime; alpha stays solid. Off-cycle
        // instances collapse to degenerate quads in the vertex shader and
        // never reach this stage.
        gl_FragColor = vec4(uDashColor, 1.0);
    }
`;
