// Shared navigation state for the slice + the constants that describe
// the navigable world it pans over. Lives outside React state because
// it's read on every frame by R3F-side components (Ground, Seagrass,
// Water) and by the requestAnimationFrame loop driving the minimap.

// Half-side of the navigable world square, in noise-world units. The
// slice's centre is clamped so the visible TANK_SIZE×TANK_SIZE window
// always fits fully inside this rectangle.
export const WORLD_HALF_EXTENT = 30;

// Side length of the visible slice ("tank") in world units. Single
// source of truth — Scene composition imports this so Ground / Seagrass
// / Water all agree on what "the slice" means.
export const TANK_SIZE = 18;

// (x, z) anchor of the slice in noise-world coords. Each frame
// Navigation writes the WASD-driven offset here and Ground / Seagrass /
// Water / Minimap read it.
export const worldOffset = { x: 0, z: 0 };

// Unit vector in the XZ plane along the camera's forward direction.
// Navigation writes this from camera.getWorldDirection so the minimap
// can draw a heading indicator without holding a reference to the
// camera object. Initialised to -z so the first frame before Navigation
// runs has a sensible default.
export const cameraFacing = { x: 0, z: -1 };
