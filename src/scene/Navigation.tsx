import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import {
    cameraFacing,
    TANK_SIZE,
    WORLD_HALF_EXTENT,
    worldOffset,
} from "./worldOffset";

// World units per second the slice's noise-space anchor advances at.
const SPEED = 8;

// Maximum |offset| in either axis. Keeps the visible slice fully inside
// the WORLD_HALF_EXTENT square — the slice's far edge reaches the world
// boundary exactly when the offset hits this value.
const MAX_OFFSET = WORLD_HALF_EXTENT - TANK_SIZE / 2;

// Pans the visible slice across the larger procedural terrain. Keys are
// view-relative: w/s move into/out of the screen along the camera's
// horizontal forward direction; a/d strafe along its right. The camera
// itself doesn't move — only the (x, z) offset fed into sampleTerrain.
export default function Navigation() {
    const { camera } = useThree();
    const held = useRef<Record<string, boolean>>({});
    const _forward = useRef(new THREE.Vector3());

    useEffect(() => {
        const onDown = (e: KeyboardEvent) => {
            if (e.metaKey || e.ctrlKey || e.altKey) return;
            const k = e.key.toLowerCase();
            if (k === "w" || k === "a" || k === "s" || k === "d") {
                held.current[k] = true;
            }
        };
        const onUp = (e: KeyboardEvent) => {
            const k = e.key.toLowerCase();
            held.current[k] = false;
        };
        const onBlur = () => {
            held.current = {};
        };
        window.addEventListener("keydown", onDown);
        window.addEventListener("keyup", onUp);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onDown);
            window.removeEventListener("keyup", onUp);
            window.removeEventListener("blur", onBlur);
        };
    }, []);

    // Negative priority so the offset and camera-facing are up to date
    // by the time Ground / Seagrass / Water read them in their default-
    // priority frame callbacks.
    useFrame((_, delta) => {
        // Project the camera's forward direction onto the XZ plane and
        // publish it for the minimap. Done every frame (not gated on
        // WASD input) so the heading indicator stays current while the
        // user is only orbiting the camera.
        camera.getWorldDirection(_forward.current);
        const fx = _forward.current.x;
        const fz = _forward.current.z;
        const flen = Math.hypot(fx, fz) || 1;
        const fwdX = fx / flen;
        const fwdZ = fz / flen;
        cameraFacing.x = fwdX;
        cameraFacing.z = fwdZ;

        const h = held.current;
        let fwdAxis = (h.w ? 1 : 0) - (h.s ? 1 : 0);
        let rightAxis = (h.d ? 1 : 0) - (h.a ? 1 : 0);
        if (fwdAxis === 0 && rightAxis === 0) return;
        // Normalize so diagonals don't move faster than cardinals.
        const alen = Math.hypot(fwdAxis, rightAxis);
        fwdAxis /= alen;
        rightAxis /= alen;

        // Right vector is forward × up (= +y), which works out to
        // (-fwdZ, +fwdX) in the XZ plane.
        const rightX = -fwdZ;
        const rightZ = fwdX;

        const dx = fwdAxis * fwdX + rightAxis * rightX;
        const dz = fwdAxis * fwdZ + rightAxis * rightZ;

        worldOffset.x = THREE.MathUtils.clamp(
            worldOffset.x + dx * SPEED * delta,
            -MAX_OFFSET,
            MAX_OFFSET,
        );
        worldOffset.z = THREE.MathUtils.clamp(
            worldOffset.z + dz * SPEED * delta,
            -MAX_OFFSET,
            MAX_OFFSET,
        );
    }, -1);

    return null;
}
