import { useEffect, useRef } from "react";
import { sampleTerrain } from "./terrain";
import {
    cameraFacing,
    TANK_SIZE,
    WORLD_HALF_EXTENT,
    worldOffset,
} from "./worldOffset";
import { WATER_Y } from "./Scene";

// Display + native canvas size (square). The same value is used for the
// HTML5 canvas resolution and the CSS size, so canvas pixels match
// display pixels 1:1 — strokes stay crisp.
const MINIMAP_SIZE = 160;

// Resolution of the pre-rendered terrain heightmap. Chosen so the
// upscale to MINIMAP_SIZE is an exact integer (160 / 80 = 2), giving
// chunky-but-clean nearest-neighbour pixels that match the scene's
// pixel-art aesthetic.
const TERRAIN_RES = 80;

// Height (in noise-world units) at which water meets land in the scene.
const WATERLINE = WATER_Y;

// Colour ramp anchors. ABOVE_WATER matches the seabed sand tone so the
// land/island parts of the minimap read in the same family as the 3D
// scene; SHALLOW / DEEP bracket the underwater range.
type RGB = { r: number; g: number; b: number };
const ABOVE_WATER: RGB = { r: 194, g: 168, b: 120 };
const SHALLOW: RGB = { r: 110, g: 200, b: 200 };
const DEEP: RGB = { r: 30, g: 60, b: 90 };

// Deepest height to span in the underwater ramp (anything lower clamps
// to DEEP). Matches the basin elevation declared in terrain.ts.
const BASIN_FLOOR = -0.5;

function heightToColor(h: number): RGB {
    if (h >= WATERLINE) return ABOVE_WATER;
    const t = Math.min(
        1,
        Math.max(0, (WATERLINE - h) / (WATERLINE - BASIN_FLOOR)),
    );
    return {
        r: Math.round(SHALLOW.r + (DEEP.r - SHALLOW.r) * t),
        g: Math.round(SHALLOW.g + (DEEP.g - SHALLOW.g) * t),
        b: Math.round(SHALLOW.b + (DEEP.b - SHALLOW.b) * t),
    };
}

// Top-down view of the navigable world with the visible slice's
// position and heading drawn on top. Lives outside the R3F Canvas as a
// regular DOM element — pointerEvents: none so it doesn't intercept
// drag-to-orbit input directed at the scene below.
export default function Minimap() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const terrainCanvasRef = useRef<HTMLCanvasElement | null>(null);

    // One-shot terrain pre-render. Sampling sampleTerrain at TERRAIN_RES²
    // points is ~6k fbm evaluations — done once on mount, cached as a
    // small offscreen canvas, and stamped onto the visible canvas each
    // frame via drawImage (nearest-neighbour upscale).
    useEffect(() => {
        const off = document.createElement("canvas");
        off.width = TERRAIN_RES;
        off.height = TERRAIN_RES;
        const ctx = off.getContext("2d");
        if (!ctx) return;
        const image = ctx.createImageData(TERRAIN_RES, TERRAIN_RES);
        const worldExtent = WORLD_HALF_EXTENT * 2;
        for (let py = 0; py < TERRAIN_RES; py++) {
            for (let px = 0; px < TERRAIN_RES; px++) {
                const wx = ((px + 0.5) / TERRAIN_RES - 0.5) * worldExtent;
                const wz = ((py + 0.5) / TERRAIN_RES - 0.5) * worldExtent;
                const { height } = sampleTerrain(wx, wz);
                const c = heightToColor(height);
                const idx = (py * TERRAIN_RES + px) * 4;
                image.data[idx + 0] = c.r;
                image.data[idx + 1] = c.g;
                image.data[idx + 2] = c.b;
                image.data[idx + 3] = 255;
            }
        }
        ctx.putImageData(image, 0, 0);
        terrainCanvasRef.current = off;
    }, []);

    // requestAnimationFrame loop. The minimap can't piggyback on R3F's
    // useFrame because it isn't inside the Canvas tree; raf gives us a
    // synced redraw without re-rendering the React component.
    useEffect(() => {
        let raf = 0;
        const draw = () => {
            const canvas = canvasRef.current;
            const terrain = terrainCanvasRef.current;
            if (canvas && terrain) {
                const ctx = canvas.getContext("2d");
                if (ctx) {
                    ctx.imageSmoothingEnabled = false;
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(
                        terrain,
                        0,
                        0,
                        terrain.width,
                        terrain.height,
                        0,
                        0,
                        canvas.width,
                        canvas.height,
                    );

                    // Map world XZ → canvas pixels. World 0,0 sits at
                    // the canvas centre; +z runs down to match the
                    // top-down convention (north-up == -z).
                    const worldExtent = WORLD_HALF_EXTENT * 2;
                    const scale = canvas.width / worldExtent;
                    const cx = (worldOffset.x + WORLD_HALF_EXTENT) * scale;
                    const cz = (worldOffset.z + WORLD_HALF_EXTENT) * scale;
                    const sliceW = TANK_SIZE * scale;

                    // Slice outline — axis-aligned because the world
                    // window itself is axis-aligned regardless of how
                    // the camera is rotated around it.
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = "#ffffff";
                    ctx.strokeRect(
                        cx - sliceW / 2,
                        cz - sliceW / 2,
                        sliceW,
                        sliceW,
                    );

                    // Heading arrow — short stub showing which way is
                    // "forward" for WASD, since that direction tracks
                    // the camera's orbit.
                    const len = sliceW * 0.45;
                    ctx.beginPath();
                    ctx.moveTo(cx, cz);
                    ctx.lineTo(
                        cx + cameraFacing.x * len,
                        cz + cameraFacing.z * len,
                    );
                    ctx.strokeStyle = "#ffd64a";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            }
            raf = requestAnimationFrame(draw);
        };
        raf = requestAnimationFrame(draw);
        return () => cancelAnimationFrame(raf);
    }, []);

    return (
        <canvas
            ref={canvasRef}
            width={MINIMAP_SIZE}
            height={MINIMAP_SIZE}
            style={{
                position: "absolute",
                top: 16,
                right: 16,
                width: MINIMAP_SIZE,
                height: MINIMAP_SIZE,
                imageRendering: "pixelated",
                outline: "2px solid #ffffff",
                pointerEvents: "none",
                background: "#000",
            }}
        />
    );
}
