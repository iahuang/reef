import { Canvas } from "@react-three/fiber";
import { PerspectiveCamera } from "@react-three/drei";
import { useCallback, useEffect, useRef, useState } from "react";
import Scene from "./scene/Scene";
import CameraRig from "./scene/CameraRig";
import Navigation from "./scene/Navigation";
import Minimap from "./scene/Minimap";

// Forcing a low device pixel ratio renders the scene at a fraction of the
// canvas's CSS pixel size; combined with `image-rendering: pixelated` on the
// <canvas> element the result reads as chunky pixel-art blocks.
const PIXEL_DPR = 2 / window.devicePixelRatio;

// Discard the first N frames after a population change before sampling.
// Fish spawn at the surface and need a moment for the terrain force to
// settle them — measuring before then biases the average toward the
// transient.
const WARMUP_FRAMES = 30;

const INITIAL_FISH_COUNT = 60;

export default function App() {
    const [fishCount, setFishCount] = useState(INITIAL_FISH_COUNT);
    const [currentAvg, setCurrentAvg] = useState(0);
    const [currentSamples, setCurrentSamples] = useState(0);
    const [history, setHistory] = useState<{ count: number; avgMs: number }[]>(
        [],
    );

    // Per-frame accumulator. Kept in a ref so useFrame can mutate it
    // without triggering React renders sixty times per second.
    const timingRef = useRef({ total: 0, samples: 0, skip: WARMUP_FRAMES });

    useEffect(() => {
        timingRef.current = { total: 0, samples: 0, skip: WARMUP_FRAMES };
        setCurrentAvg(0);
        setCurrentSamples(0);
    }, [fishCount]);

    // Pull from the ref into React state on a slow tick so the overlay
    // updates a few times per second without re-rendering each frame.
    useEffect(() => {
        const id = setInterval(() => {
            const t = timingRef.current;
            if (t.samples > 0) {
                setCurrentAvg(t.total / t.samples);
                setCurrentSamples(t.samples);
            }
        }, 250);
        return () => clearInterval(id);
    }, []);

    const onFishSample = useCallback((ms: number) => {
        const t = timingRef.current;
        if (t.skip > 0) {
            t.skip--;
            return;
        }
        t.total += ms;
        t.samples++;
    }, []);

    const handleAddFish = () => {
        const t = timingRef.current;
        if (t.samples > 0) {
            setHistory((prev) => [
                ...prev,
                { count: fishCount, avgMs: t.total / t.samples },
            ]);
        }
        // +50%, rounding up. Math.max guards the early steps so we
        // always grow by at least one fish even if Math.ceil rounds to
        // the same value.
        setFishCount((c) => Math.max(c + 1, Math.ceil(c * 1.5)));
    };

    const handleReset = () => {
        setHistory([]);
        setFishCount(INITIAL_FISH_COUNT);
    };

    return (
        <div className="w-screen h-screen bg-sky-300 relative">
            <Canvas dpr={[PIXEL_DPR, PIXEL_DPR]} gl={{ antialias: false }} flat>
                <color attach="background" args={["#9fdcec"]} />
                <PerspectiveCamera
                    makeDefault
                    near={0.1}
                    far={300}
                    fov={35}
                />
                <CameraRig
                    initialAzimuth={Math.PI / 4}
                    initialPolar={0.9}
                    initialDistance={50}
                    azimuthSteps={16}
                    polarSteps={8}
                    minDistance={20}
                    maxDistance={150}
                />
                <Navigation />
                <Scene fishCount={fishCount} onFishSample={onFishSample} />
            </Canvas>
            <Minimap />
            <div
                style={{
                    position: "absolute",
                    top: 16,
                    left: 16,
                    fontFamily: "monospace",
                    fontSize: 12,
                    color: "#fff",
                    background: "rgba(0, 0, 0, 0.65)",
                    padding: "10px 12px",
                    borderRadius: 4,
                    minWidth: 200,
                    lineHeight: 1.5,
                    userSelect: "none",
                }}
            >
                <div>
                    Fish: <b>{fishCount}</b>
                </div>
                <div>
                    Avg:{" "}
                    <b>
                        {currentSamples > 0 ? currentAvg.toFixed(2) : "—"} ms
                    </b>
                    <span style={{ opacity: 0.6 }}>
                        {" "}
                        ({currentSamples} samples)
                    </span>
                </div>
                <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                    <button
                        onClick={handleAddFish}
                        style={{
                            flex: 1,
                            padding: "6px 8px",
                            background: "#ffd64a",
                            color: "#000",
                            border: "none",
                            borderRadius: 3,
                            cursor: "pointer",
                            fontFamily: "monospace",
                            fontWeight: "bold",
                        }}
                    >
                        Spawn +50%
                    </button>
                    <button
                        onClick={handleReset}
                        style={{
                            padding: "6px 8px",
                            background: "#444",
                            color: "#fff",
                            border: "none",
                            borderRadius: 3,
                            cursor: "pointer",
                            fontFamily: "monospace",
                        }}
                    >
                        Reset
                    </button>
                </div>
                {history.length > 0 && (
                    <div
                        style={{
                            marginTop: 10,
                            paddingTop: 8,
                            borderTop: "1px solid #555",
                        }}
                    >
                        {history.map((h, i) => (
                            <div
                                key={i}
                                style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                }}
                            >
                                <span style={{ opacity: 0.8 }}>{h.count}</span>
                                <span>{h.avgMs.toFixed(2)} ms</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
