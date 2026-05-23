import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { sampleTerrain } from "./terrain";
import { worldOffset, WORLD_HALF_EXTENT } from "./worldOffset";

// --- Boids tuning ---
// Distance within which a fish counts another as a neighbour for
// alignment / cohesion.
const PERCEPTION = 2.0;
// Inside this, the separation force kicks in to prevent stacking.
const SEPARATION = 0.55;
// Cruise speed bounds (world units / second). MIN keeps fish from
// stalling into a clump and stopping; MAX caps the integration step.
const MAX_SPEED = 1.6;
const MIN_SPEED = 0.7;
// Per-frame steering ceiling. Higher = more responsive but jitter-prone.
const MAX_FORCE = 3.5;

// Force weights. Tuned so cohesion + alignment dominate at mid range and
// separation only spikes at close encounters; the school stays loose but
// coherent rather than collapsing to a point or spraying apart.
const W_SEPARATION = 1.6;
const W_ALIGNMENT = 1.0;
const W_COHESION = 0.7;
const W_BOUNDS = 3.0;

// Hard wall and margin (noise-world units). Fish steer back when they
// enter the margin band — gives curved turn-around rather than slamming
// the limit.
const XZ_BOUND = WORLD_HALF_EXTENT - 0.5;
const XZ_MARGIN = 4.0;

// --- Spatial hash (uniform XZ grid) ---
// Cell side equals PERCEPTION, so any pair within range must share a
// cell or be in two diagonally-adjacent cells. The neighbour scan
// therefore visits a fixed 3×3 footprint per fish instead of all N.
// We grid in XZ only — the fish swim in a thin Y layer (~5 units), so
// adding a Y axis would 3× the cell-iteration count for no real benefit
// in candidate culling.
const CELL_SIZE = PERCEPTION;
const GRID = Math.ceil((WORLD_HALF_EXTENT * 2) / CELL_SIZE);
const CELL_COUNT = GRID * GRID;

// --- Terrain avoidance ---
// Vertical clearance the fish tries to keep above the seabed. Tuned to
// be a bit more than the body's longest half-extent so the mesh never
// visibly grazes the ground.
const TERRAIN_CLEARANCE = 0.55;
// Distance along the horizontal velocity to probe for rising terrain.
// Long enough that a fish can climb in time at MAX_SPEED + MAX_FORCE,
// short enough that they don't bail out of a basin too far in advance.
const TERRAIN_LOOKAHEAD = 0.9;
// Restoring forces when the look-ahead or here-probe is too close to
// the seabed. UP lifts straight up; REDIRECT steers horizontal
// velocity toward the local downhill direction so the fish swims
// around steep slopes instead of climbing them. The older "brake"
// form (damping horizontal speed) stalled fish at the foot of cliffs
// and could pin them between rising terrain and the water surface.
const W_TERRAIN_UP = 6.0;
const W_TERRAIN_REDIRECT = 3.0;
// Finite-difference step for the in-plane terrain gradient. Large
// enough to average over DETAIL_FREQUENCY noise (period ≈ 0.4) but
// small relative to the large-feature scale (NOISE_FREQUENCY period
// ≈ 12.5), so the gradient reflects local slope rather than ripples.
const TERRAIN_GRAD_EPS = 1.0;

// Module-scope scratch for steerToward so the inner loop allocates no
// arrays. Returned by reference; callers must copy out before the next
// steerToward call.
const _steer: [number, number, number] = [0, 0, 0];

// Reynolds-style "steering" force: desired velocity (along the target
// direction at MAX_SPEED) minus current velocity, clamped to MAX_FORCE.
// Empty input → zero output.
function steerToward(
    dx: number, dy: number, dz: number,
    vx: number, vy: number, vz: number,
): [number, number, number] {
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) {
        _steer[0] = 0; _steer[1] = 0; _steer[2] = 0;
        return _steer;
    }
    let sx = (dx / len) * MAX_SPEED - vx;
    let sy = (dy / len) * MAX_SPEED - vy;
    let sz = (dz / len) * MAX_SPEED - vz;
    const slen = Math.hypot(sx, sy, sz);
    if (slen > MAX_FORCE) {
        const c = MAX_FORCE / slen;
        sx *= c; sy *= c; sz *= c;
    }
    _steer[0] = sx; _steer[1] = sy; _steer[2] = sz;
    return _steer;
}

type Props = {
    // Water-surface Y in the scene group's local frame. Fish swim in a
    // 2-unit-thick layer just below this — keeps them clearly underwater
    // without needing per-fish terrain queries to dodge the seabed.
    surfaceY: number;
    // Number of boids in the school. Lifted to a prop so the parent can
    // grow the population at runtime for limit-testing.
    count: number;
    // Optional per-frame callback receiving the wall-clock ms spent in
    // the simulation + render pass. Used by the benchmark overlay.
    onSample?: (ms: number) => void;
};

export default function Fish({ surfaceY, count, onSample }: Props) {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);
    const targetVec = useMemo(() => new THREE.Vector3(), []);

    // yMax is just below the water surface — a soft ceiling so fish
    // don't breach. The floor is handled per-fish by terrain look-ahead,
    // not a fixed Y, so fish can dive into basins and rise over reefs.
    const yMax = surfaceY - 0.3;
    // Hard safety clamp on Y. Fish can never integrate below this no
    // matter what — guards against pathological force sums.
    const yFloor = -1.0;

    // Position + velocity arrays in noise-world coords. SoA (parallel
    // Float32Arrays) instead of an array of objects so the per-frame
    // inner loop touches contiguous memory with no allocations. Rebuilt
    // when `count` changes so the school resets to a known starting
    // configuration at each step of a limit-test sweep.
    const state = useMemo(() => {
        const pos = new Float32Array(count * 3);
        const vel = new Float32Array(count * 3);
        // Spread spawns across the navigable world (inside the soft-
        // margin band) so the population fragments into many small
        // schools instead of merging into one giant cluster — the
        // spatial hash then sees low per-cell occupancy throughout the
        // benchmark window. Y is sampled in the water column above
        // local terrain; if the chosen XZ lands on an island peak, a
        // few rejection retries find open water.
        const spawnHalf = XZ_BOUND - XZ_MARGIN;
        const ceilY = yMax - 0.3;
        for (let i = 0; i < count; i++) {
            let x = 0, z = 0, terrainH = 0;
            for (let attempt = 0; attempt < 4; attempt++) {
                x = (Math.random() - 0.5) * 2 * spawnHalf;
                z = (Math.random() - 0.5) * 2 * spawnHalf;
                terrainH = sampleTerrain(x, z).height;
                if (terrainH < yMax - 1.0) break;
            }
            const floorY = terrainH + TERRAIN_CLEARANCE + 0.1;
            const y =
                floorY < ceilY
                    ? floorY + Math.random() * (ceilY - floorY)
                    : ceilY - 0.05;
            pos[i * 3 + 0] = x;
            pos[i * 3 + 1] = y;
            pos[i * 3 + 2] = z;
            const angle = Math.random() * Math.PI * 2;
            vel[i * 3 + 0] = Math.cos(angle) * MAX_SPEED;
            vel[i * 3 + 1] = (Math.random() - 0.5) * 0.2;
            vel[i * 3 + 2] = Math.sin(angle) * MAX_SPEED;
        }
        return { pos, vel };
    }, [yMax, count]);

    // Fixed-size grid bookkeeping. Counts and the prefix-sum array
    // depend only on the grid resolution, so they're allocated once.
    const grid = useMemo(
        () => ({
            cellCount: new Int32Array(CELL_COUNT),
            cellStart: new Int32Array(CELL_COUNT + 1),
        }),
        [],
    );

    // Per-fish grid buffers: which cell each fish lives in, and an
    // index list sorted by cell so all members of a cell are contiguous.
    const gridIndex = useMemo(
        () => ({
            fishCell: new Int32Array(count),
            cellOrder: new Int32Array(count),
        }),
        [count],
    );

    // Primitive fish shape: a stretched octahedron whose long axis is
    // along z. Object3D.lookAt aligns the object's -z to the target, so
    // we steer with target = position + velocity and the long axis falls
    // into the direction of motion. Symmetric end-to-end — head and
    // tail look the same, but at this scale it reads fine.
    const geometry = useMemo(() => {
        const g = new THREE.OctahedronGeometry(0.18);
        g.scale(0.55, 0.4, 1.7);
        return g;
    }, []);

    useFrame((_, delta) => {
        const mesh = meshRef.current;
        if (!mesh) return;
        const t0 = performance.now();
        const { pos, vel } = state;
        // Clamp dt so a long pause (tab backgrounded, breakpoint, etc.)
        // can't deliver a 5-second step that explodes the simulation.
        const dt = Math.min(delta, 0.05);

        const percSq = PERCEPTION * PERCEPTION;
        const sepSq = SEPARATION * SEPARATION;

        // ---- Build XZ uniform grid (counting-sort layout) ----
        // Pass 1: compute each fish's cell and tally cell occupancy.
        // Pass 2: prefix sum into cellStart so cell c owns the slice
        //         cellOrder[cellStart[c] .. cellStart[c+1]].
        // Pass 3: scatter fish indices into cellOrder using cellCount
        //         as a write cursor.
        const { cellCount, cellStart } = grid;
        const { fishCell, cellOrder } = gridIndex;
        cellCount.fill(0);
        for (let i = 0; i < count; i++) {
            const ix3 = i * 3;
            let cx = ((pos[ix3] + WORLD_HALF_EXTENT) / CELL_SIZE) | 0;
            let cz = ((pos[ix3 + 2] + WORLD_HALF_EXTENT) / CELL_SIZE) | 0;
            // Defensive clamp: a fish at exactly +XZ_BOUND or driven
            // briefly outside by the soft margin force could otherwise
            // index past the grid.
            if (cx < 0) cx = 0; else if (cx >= GRID) cx = GRID - 1;
            if (cz < 0) cz = 0; else if (cz >= GRID) cz = GRID - 1;
            const c = cx * GRID + cz;
            fishCell[i] = c;
            cellCount[c]++;
        }
        cellStart[0] = 0;
        for (let c = 0; c < CELL_COUNT; c++) {
            cellStart[c + 1] = cellStart[c] + cellCount[c];
        }
        cellCount.fill(0);
        for (let i = 0; i < count; i++) {
            const c = fishCell[i];
            cellOrder[cellStart[c] + cellCount[c]++] = i;
        }

        // ---- Boids update: 3×3 cell neighbour scan + integration ----
        // Per-fish candidate count is ~9 × (count / CELL_COUNT), so the
        // whole pass scales with N at fixed world size instead of N².
        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            const px = pos[ix], py = pos[ix + 1], pz = pos[ix + 2];
            const vx = vel[ix], vy = vel[ix + 1], vz = vel[ix + 2];

            let sepX = 0, sepY = 0, sepZ = 0;
            let aliX = 0, aliY = 0, aliZ = 0;
            let cohX = 0, cohY = 0, cohZ = 0;
            let neighbors = 0;
            let separators = 0;

            // Walk the 3×3 cell footprint around the fish's own cell.
            // Cell side = PERCEPTION, so any j with distSq < percSq is
            // guaranteed to live in one of these nine cells.
            const myCell = fishCell[i];
            const myCx = (myCell / GRID) | 0;
            const myCz = myCell - myCx * GRID;
            const cxMin = myCx > 0 ? myCx - 1 : 0;
            const cxMax = myCx < GRID - 1 ? myCx + 1 : GRID - 1;
            const czMin = myCz > 0 ? myCz - 1 : 0;
            const czMax = myCz < GRID - 1 ? myCz + 1 : GRID - 1;

            for (let cx = cxMin; cx <= cxMax; cx++) {
                const rowBase = cx * GRID;
                for (let cz = czMin; cz <= czMax; cz++) {
                    const c = rowBase + cz;
                    const kStart = cellStart[c];
                    const kEnd = cellStart[c + 1];
                    for (let k = kStart; k < kEnd; k++) {
                        const j = cellOrder[k];
                        if (j === i) continue;
                        const jx = j * 3;
                        const dx = pos[jx] - px;
                        const dy = pos[jx + 1] - py;
                        const dz = pos[jx + 2] - pz;
                        const distSq = dx * dx + dy * dy + dz * dz;
                        if (distSq < percSq) {
                            neighbors++;
                            aliX += vel[jx];
                            aliY += vel[jx + 1];
                            aliZ += vel[jx + 2];
                            cohX += pos[jx];
                            cohY += pos[jx + 1];
                            cohZ += pos[jx + 2];
                            if (distSq < sepSq && distSq > 1e-6) {
                                // Separation force weighted by 1/dist
                                // so close neighbours push much harder
                                // than mid-range ones — the standard
                                // Reynolds form.
                                const inv = 1.0 / Math.sqrt(distSq);
                                sepX -= dx * inv;
                                sepY -= dy * inv;
                                sepZ -= dz * inv;
                                separators++;
                            }
                        }
                    }
                }
            }

            let ax = 0, ay = 0, az = 0;

            if (separators > 0) {
                const s = steerToward(sepX, sepY, sepZ, vx, vy, vz);
                ax += s[0] * W_SEPARATION;
                ay += s[1] * W_SEPARATION;
                az += s[2] * W_SEPARATION;
            }
            if (neighbors > 0) {
                const a = steerToward(aliX, aliY, aliZ, vx, vy, vz);
                ax += a[0] * W_ALIGNMENT;
                ay += a[1] * W_ALIGNMENT;
                az += a[2] * W_ALIGNMENT;

                const cxd = cohX / neighbors - px;
                const cyd = cohY / neighbors - py;
                const czd = cohZ / neighbors - pz;
                const c = steerToward(cxd, cyd, czd, vx, vy, vz);
                ax += c[0] * W_COHESION;
                ay += c[1] * W_COHESION;
                az += c[2] * W_COHESION;
            }

            // Soft XZ wall repulsion. Linear push back when inside the
            // margin band so motion curves rather than slamming the limit.
            if (px > XZ_BOUND - XZ_MARGIN) ax -= (px - (XZ_BOUND - XZ_MARGIN)) * W_BOUNDS;
            if (px < -XZ_BOUND + XZ_MARGIN) ax += ((-XZ_BOUND + XZ_MARGIN) - px) * W_BOUNDS;
            if (pz > XZ_BOUND - XZ_MARGIN) az -= (pz - (XZ_BOUND - XZ_MARGIN)) * W_BOUNDS;
            if (pz < -XZ_BOUND + XZ_MARGIN) az += ((-XZ_BOUND + XZ_MARGIN) - pz) * W_BOUNDS;
            // Soft surface ceiling. 2× weight because the breach would
            // be more visually wrong than the floor case (the floor is
            // handled below by terrain look-ahead).
            if (py > yMax - 0.3) ay -= (py - (yMax - 0.3)) * W_BOUNDS * 2.0;

            // ---- Terrain avoidance ----
            // Sample two points: directly under the fish (catches sharp
            // pop-up clipping) and one step ahead along velocity (gives
            // anticipation so the fish climbs before it would have hit).
            // Use the higher of the two as the effective seabed and push
            // up if the fish's Y is below seabed + clearance.
            const speedXZ = Math.hypot(vx, vz);
            let aheadX = px;
            let aheadZ = pz;
            if (speedXZ > 1e-3) {
                aheadX = px + (vx / speedXZ) * TERRAIN_LOOKAHEAD;
                aheadZ = pz + (vz / speedXZ) * TERRAIN_LOOKAHEAD;
            }
            const gHere = sampleTerrain(px, pz).height;
            const gAhead = sampleTerrain(aheadX, aheadZ).height;
            const ceiling =
                Math.max(gHere, gAhead) + TERRAIN_CLEARANCE;
            if (py < ceiling) {
                const overlap = ceiling - py;
                ay += overlap * W_TERRAIN_UP;
                // Horizontal redirect along the local downhill. Two
                // extra terrain samples; gated by py < ceiling so fish
                // in open water pay nothing.
                const gPlusX = sampleTerrain(px + TERRAIN_GRAD_EPS, pz).height;
                const gPlusZ = sampleTerrain(px, pz + TERRAIN_GRAD_EPS).height;
                const gradX = (gPlusX - gHere) / TERRAIN_GRAD_EPS;
                const gradZ = (gPlusZ - gHere) / TERRAIN_GRAD_EPS;
                if (gradX * gradX + gradZ * gradZ > 1e-6) {
                    // Steer horizontally toward -gradient (downhill).
                    // Feed vy = 0 so the returned force has no Y
                    // component — vertical is owned by W_TERRAIN_UP.
                    const s = steerToward(-gradX, 0, -gradZ, vx, 0, vz);
                    ax += s[0] * W_TERRAIN_REDIRECT;
                    az += s[2] * W_TERRAIN_REDIRECT;
                }
            }

            // Integrate velocity, then clamp speed to [MIN, MAX].
            let nvx = vx + ax * dt;
            let nvy = vy + ay * dt;
            let nvz = vz + az * dt;
            const speed = Math.hypot(nvx, nvy, nvz);
            if (speed > MAX_SPEED) {
                const c = MAX_SPEED / speed;
                nvx *= c; nvy *= c; nvz *= c;
            } else if (speed < MIN_SPEED && speed > 1e-6) {
                const c = MIN_SPEED / speed;
                nvx *= c; nvy *= c; nvz *= c;
            }
            vel[ix] = nvx;
            vel[ix + 1] = nvy;
            vel[ix + 2] = nvz;

            pos[ix] += nvx * dt;
            pos[ix + 1] += nvy * dt;
            pos[ix + 2] += nvz * dt;

            // Hard clamp safety net. The soft margin force usually
            // catches fish first, but a fast diagonal approach can
            // outrun it for a frame.
            if (pos[ix] > XZ_BOUND) pos[ix] = XZ_BOUND;
            if (pos[ix] < -XZ_BOUND) pos[ix] = -XZ_BOUND;
            if (pos[ix + 2] > XZ_BOUND) pos[ix + 2] = XZ_BOUND;
            if (pos[ix + 2] < -XZ_BOUND) pos[ix + 2] = -XZ_BOUND;
            if (pos[ix + 1] > yMax) pos[ix + 1] = yMax;
            if (pos[ix + 1] < yFloor) pos[ix + 1] = yFloor;
        }

        // ---- Render pass: world → scene-local + orient to velocity ----
        for (let i = 0; i < count; i++) {
            const ix = i * 3;
            const lx = pos[ix] - worldOffset.x;
            const ly = pos[ix + 1];
            const lz = pos[ix + 2] - worldOffset.z;
            dummy.position.set(lx, ly, lz);
            targetVec.set(
                lx + vel[ix],
                ly + vel[ix + 1],
                lz + vel[ix + 2],
            );
            dummy.lookAt(targetVec);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (onSample) onSample(performance.now() - t0);
    });

    return (
        <instancedMesh
            ref={meshRef}
            // Key on count so a population change tears down the
            // fixed-size InstancedMesh buffer and allocates a new one.
            key={count}
            args={[geometry, undefined, count]}
            frustumCulled={false}
        >
            <meshStandardMaterial
                color="#e07e58"
                roughness={0.7}
                metalness={0.05}
                flatShading
            />
        </instancedMesh>
    );
}
