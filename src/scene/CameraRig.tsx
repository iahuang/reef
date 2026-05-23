import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

type Props = {
    initialAzimuth?: number;
    initialPolar?: number;
    initialDistance?: number;
    initialZoom?: number;
    azimuthSteps?: number;
    polarSteps?: number;
    minDistance?: number;
    maxDistance?: number;
    minZoom?: number;
    maxZoom?: number;
    minPolar?: number;
    maxPolar?: number;
    // Exponential rate for the tween between snapped stops. Higher = snappier
    // (closer to a hard snap); lower = more glide. Around 18–25 gives a clean
    // tick. Set to a very large number (e.g. 1000) to recover the instant snap.
    tweenRate?: number;
};

// Drag rotates the camera, wheel zooms. Azimuth and polar angles are
// quantised to a fixed number of discrete stops, so within a single stop
// every edge in the scene lands on exactly the same low-resolution pixels
// frame after frame — eliminating the rotational pixel crawl that a
// continuously-rotating camera produces on a downsampled framebuffer.
export default function CameraRig({
    initialAzimuth = Math.PI / 4,
    initialPolar = 0.9,
    initialDistance = 30,
    initialZoom = 45,
    azimuthSteps = 16,
    polarSteps = 6,
    minDistance = 10,
    maxDistance = 80,
    minZoom = 15,
    maxZoom = 200,
    minPolar = 0.2,
    maxPolar = Math.PI / 2 - 0.08,
    tweenRate = 20,
}: Props) {
    const { camera, gl } = useThree();
    // Target (snapped) values that input writes to.
    const azimuth = useRef(initialAzimuth);
    const polar = useRef(initialPolar);
    const distance = useRef(initialDistance);
    const zoom = useRef(initialZoom);
    // Displayed values that the camera actually uses; each frame they ease
    // toward the snapped targets. Camera rendering only sees these smoothed
    // values, so motion between stops is a continuous glide.
    const displayedAzimuth = useRef(initialAzimuth);
    const displayedPolar = useRef(initialPolar);
    const target = useRef(new THREE.Vector3(0, 0, 0));

    useEffect(() => {
        const el = gl.domElement;
        let isDragging = false;
        let lastX = 0;
        let lastY = 0;
        // Unquantised drag accumulator. The drag updates this freely and the
        // quantised refs are derived from it on every move; this way slow
        // drags advance one step at a time instead of jumping multiple stops.
        let rawAzimuth = azimuth.current;
        let rawPolar = polar.current;

        const azStep = (Math.PI * 2) / azimuthSteps;
        const polarRange = maxPolar - minPolar;
        const polarStep =
            polarSteps > 1 ? polarRange / (polarSteps - 1) : polarRange;

        const snapAz = (v: number) => Math.round(v / azStep) * azStep;
        const snapPolar = (v: number) => {
            const clamped = THREE.MathUtils.clamp(v, minPolar, maxPolar);
            if (polarSteps <= 1) return minPolar + polarRange * 0.5;
            return (
                minPolar +
                Math.round((clamped - minPolar) / polarStep) * polarStep
            );
        };

        const onDown = (e: PointerEvent) => {
            isDragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
            rawAzimuth = azimuth.current;
            rawPolar = polar.current;
            el.setPointerCapture(e.pointerId);
        };
        const onMove = (e: PointerEvent) => {
            if (!isDragging) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            rawAzimuth += dx * 0.006;
            rawPolar = THREE.MathUtils.clamp(
                rawPolar - dy * 0.006,
                minPolar,
                maxPolar,
            );
            azimuth.current = snapAz(rawAzimuth);
            polar.current = snapPolar(rawPolar);
        };
        const onUp = (e: PointerEvent) => {
            isDragging = false;
            if (el.hasPointerCapture(e.pointerId)) {
                el.releasePointerCapture(e.pointerId);
            }
        };
        const onWheel = (e: WheelEvent) => {
            e.preventDefault();
            // For an orthographic camera, distance is irrelevant to apparent
            // size — we drive the camera's zoom factor instead. For a
            // perspective camera, dolly the camera along its orbital radius.
            if (camera instanceof THREE.OrthographicCamera) {
                const factor = Math.exp(-e.deltaY * 0.001);
                zoom.current = THREE.MathUtils.clamp(
                    zoom.current * factor,
                    minZoom,
                    maxZoom,
                );
            } else {
                const factor = Math.exp(e.deltaY * 0.001);
                distance.current = THREE.MathUtils.clamp(
                    distance.current * factor,
                    minDistance,
                    maxDistance,
                );
            }
        };

        el.addEventListener("pointerdown", onDown);
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        el.addEventListener("wheel", onWheel, { passive: false });
        return () => {
            el.removeEventListener("pointerdown", onDown);
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            el.removeEventListener("wheel", onWheel);
        };
    }, [
        gl,
        camera,
        azimuthSteps,
        polarSteps,
        minDistance,
        maxDistance,
        minZoom,
        maxZoom,
        minPolar,
        maxPolar,
    ]);

    useFrame((_, delta) => {
        // Frame-rate-independent exponential smoothing: the fraction of the
        // remaining gap to close this frame is 1 - e^(-rate * dt).
        const k = 1 - Math.exp(-tweenRate * delta);
        displayedAzimuth.current +=
            (azimuth.current - displayedAzimuth.current) * k;
        displayedPolar.current += (polar.current - displayedPolar.current) * k;

        const az = displayedAzimuth.current;
        const p = displayedPolar.current;
        const r = distance.current;
        const horiz = r * Math.sin(p);
        camera.position.set(
            target.current.x + horiz * Math.cos(az),
            target.current.y + r * Math.cos(p),
            target.current.z + horiz * Math.sin(az),
        );
        camera.lookAt(target.current);
        if (camera instanceof THREE.OrthographicCamera) {
            if (camera.zoom !== zoom.current) {
                camera.zoom = zoom.current;
                camera.updateProjectionMatrix();
            }
        }
    });

    return null;
}
