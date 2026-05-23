import Water from "./Water";
import Ground from "./Ground";
import Seagrass from "./Seagrass";
import Coral from "./Coral";
import BrainCoral from "./BrainCoral";
import FanCoral from "./FanCoral";
import Fish from "./Fish";
import { TANK_SIZE } from "./worldOffset";

// Absolute Ys (in this group's local frame) shared by the water and the
// terrain. WATER_Y is the surface plane; TERRAIN_BASE_Y is the underside
// of the seabed slab and the bottom edge of the water side panels — so
// the tinted volume extends down to the visible terrain rim instead of
// stopping at the dune baseline.
export const WATER_Y = 4.7;
const TERRAIN_BASE_Y = -1;

type Props = {
    fishCount: number;
    onFishSample: (ms: number) => void;
};

export default function Scene({ fishCount, onFishSample }: Props) {
    return (
        <group position={[0, -2, 0]}>
            <ambientLight intensity={0.75} color={"#cfe8ff"} />
            <directionalLight
                position={[10, 15, 8]}
                intensity={2}
                color={"#fff5e0"}
            />
            <directionalLight
                position={[-8, 6, -6]}
                intensity={0.3}
                color={"#9ed7ff"}
            />

            <Ground size={TANK_SIZE} waterY={WATER_Y} baseY={TERRAIN_BASE_Y} />
            <Seagrass size={TANK_SIZE} waterY={WATER_Y} />
            <Coral size={TANK_SIZE} waterY={WATER_Y} variantCount={6} />
            <BrainCoral size={TANK_SIZE} waterY={WATER_Y} variantCount={4} />
            <FanCoral size={TANK_SIZE} waterY={WATER_Y} variantCount={5} />
            <Fish surfaceY={WATER_Y} count={fishCount} onSample={onFishSample} />
            <Water size={TANK_SIZE} surfaceY={WATER_Y} baseY={TERRAIN_BASE_Y} />
        </group>
    );
}
