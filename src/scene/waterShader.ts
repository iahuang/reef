// Common depth-extinction code, prepended to both fragment shaders so the
// path-length / Beer–Lambert calculation is identical for the surface and
// the side panels. The corresponding uniforms are populated from Water.tsx
// and shared by reference between both materials.
const EXTINCTION_GLSL = /* glsl */ `
    uniform sampler2D uSceneDepth;
    uniform vec2 uResolution;
    uniform float uCameraNear;
    uniform float uCameraFar;
    uniform vec3 uExtinction;
    uniform vec3 uWaterTint;
    uniform float uMaxPathLength;
    uniform float uMinAlpha;
    uniform vec3 uFresnelTint;
    uniform float uFresnelStrength;

    // three.js packing.glsl helper, inlined: non-linear window-space depth
    // → view-space Z under a perspective projection. Returns negative
    // (view-Z grows toward -∞ in front of the camera).
    float perspectiveDepthToViewZ(float d, float near, float far) {
        return (near * far) / ((far - near) * d - far);
    }

    // Distance the view ray travels through water from this fragment to the
    // opaque scene fragment behind it (sampled from the depth pre-pass).
    // Under perspective the view rays diverge, so a true ray length would
    // divide |Δview-Z| by |viewDir.z|; at the current narrow FOV that
    // correction is a few percent at the frame edges and we skip it. If
    // the pre-pass had no geometry at this pixel, the sampled depth is the
    // far plane; fall back to uMaxPathLength so water against the sky
    // doesn't bake to full opacity at a hard silhouette.
    float waterPathLength(float fragClipZ) {
        vec2 sUv = gl_FragCoord.xy / uResolution;
        float sceneClipZ = texture2D(uSceneDepth, sUv).r;
        if (sceneClipZ >= 1.0 - 1e-6) {
            return uMaxPathLength;
        }
        float sceneViewZ = perspectiveDepthToViewZ(sceneClipZ, uCameraNear, uCameraFar);
        float waterViewZ = perspectiveDepthToViewZ(fragClipZ, uCameraNear, uCameraFar);
        return clamp(waterViewZ - sceneViewZ, 0.0, uMaxPathLength);
    }

    // Mean transmittance across RGB → scalar alpha for the standard
    // src*α + dst*(1-α) blend. Per-channel extinction still influences the
    // tint via uWaterTint (red dies fastest → blue-green dominates with
    // depth), but the alpha collapse loses some of the per-channel detail.
    // Good enough for stylized work; revisit with a colour pre-pass if the
    // per-channel chromatic shift matters.
    //
    // uMinAlpha floors the result so the water surface stays readable at
    // the waterline — where path length → 0 (e.g. where terrain pokes
    // through the surface), the volumetric term alone would let the water
    // fade to invisible and erase the shoreline.
    float extinctionAlpha(float d) {
        vec3 T = exp(-uExtinction * d);
        float a = 1.0 - dot(T, vec3(1.0 / 3.0));
        return max(a, uMinAlpha);
    }

    // Cheap Fresnel proxy: shift the water tint based on how head-on we
    // view the surface. Approximated as the Z component of the view-space
    // normal — exact under ortho, an off-axis approximation under
    // perspective that's good enough at narrow FOVs. abs() so back-facing
    // panels read the same as front-facing — they share an outward normal
    // sign in world space but the view-space sign flips depending on the
    // side. The visual purpose isn't physical correctness; it's giving
    // adjacent surfaces (e.g. the top sheet vs. a side panel) different
    // shades so their shared edge actually reads.
    vec3 applyFresnelTint(vec3 baseColor, vec3 viewNormal) {
        float facing = abs(normalize(viewNormal).z);
        float fres = 1.0 - facing;
        return mix(baseColor, uFresnelTint, fres * uFresnelStrength);
    }
`;

export const waterSurfaceVertexShader = /* glsl */ `
    uniform float uTime;
    uniform float uWaveAmplitude;
    uniform float uWaveFrequency;
    uniform float uWaveEdgeFalloff;
    uniform float uSpeed;
    // (offsetX, offsetZ) in noise-world units. Same convention as the
    // ground heightfield: sample point = local position + offset, so the
    // wave field is anchored to the underlying world rather than the
    // mesh — when the slice pans, the same wave pattern slides through
    // it instead of staying glued to the visible rectangle.
    uniform vec2 uWaterOffset;

    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vViewNormal;

    float hash(vec2 p) {
        return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
    }

    float vnoise(vec2 p) {
        vec2 i = floor(p);
        vec2 f = fract(p);
        vec2 u = f * f * (3.0 - 2.0 * f);
        float a = hash(i);
        float b = hash(i + vec2(1.0, 0.0));
        float c = hash(i + vec2(0.0, 1.0));
        float d = hash(i + vec2(1.0, 1.0));
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }

    void main() {
        vUv = uv;
        // Macro normal for the Fresnel proxy: the geometry's flat normal,
        // unaffected by the wave displacement below. The bump-driven
        // highlights in the fragment shader handle micro normal variation;
        // they shouldn't bleed into the cheap-Fresnel tint shift.
        vViewNormal = normalize(normalMatrix * normal);

        float t = uTime * uSpeed;
        vec3 pos = position;
        // The plane is rotated -π/2 about X, so local +x is world +x and
        // local +y is world -z. Adding the offset with that sign convention
        // anchors the wave field to noise-world coords.
        vec2 a = vec2(pos.x + uWaterOffset.x, pos.y - uWaterOffset.y);
        float w = 0.0;
        w += sin(a.x * uWaveFrequency + t * 1.1) * 0.5;
        w += sin(a.y * uWaveFrequency * 0.8 - t * 1.3) * 0.5;
        w += vnoise(a * 1.5 + t * 0.4) * 0.6 - 0.3;
        // Ramp the displacement to zero at the perimeter so the edge of
        // the surface stays flush with the side walls — otherwise the
        // dipping waves open visible gaps along the tank rim.
        vec2 edgeDist = min(uv, 1.0 - uv);
        float edgeFalloff = smoothstep(0.0, uWaveEdgeFalloff, min(edgeDist.x, edgeDist.y));
        pos.z += w * uWaveAmplitude * edgeFalloff;

        vec4 worldPos = modelMatrix * vec4(pos, 1.0);
        vWorldPos = worldPos.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
    }
`;

export const waterSurfaceFragmentShader = /* glsl */ `
    precision highp float;
    ${EXTINCTION_GLSL}

    uniform float uTime;
    uniform vec3 uFoamColor;
    uniform vec3 uHighlightColor;
    uniform vec3 uSunDirection;
    uniform float uBumpFrequency;
    uniform float uBumpStrength;
    uniform vec2 uBumpDrift;
    uniform float uBumpEvolutionRate;
    uniform vec2 uBumpOctaveScales;
    uniform vec2 uBumpOctaveWeights;
    uniform float uGlowThreshold;
    uniform float uBrightThreshold;
    uniform float uShoreBandWidth;
    uniform float uShoreStrength;
    // (offsetX, offsetZ) in noise-world units, matched to the vertex
    // shader's uWaterOffset. Drives the bump pattern's world anchoring.
    uniform vec2 uWaterOffset;
    // Edge length of the water-surface plane in world units. Used to
    // convert the prior UV-space bump frequency into a per-world-unit
    // frequency so the visual density of highlights doesn't change.
    uniform float uPlaneSize;

    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vViewNormal;

    // 3D simplex noise — Ashima Arts / Ian McEwan implementation
    // (https://github.com/ashima/webgl-noise, MIT-licensed). Permutation-
    // free polynomial-hash variant. Compared to value noise, the gradient
    // distribution along any single axis is much smoother — which kills
    // the global "breath" pulse you get when value-noise slices linearly
    // interp through their midpoint and the gradient variance halves.
    vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
    vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
    vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

    float snoise(vec3 v) {
        const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

        vec3 i = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);

        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);

        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;

        i = mod289(i);
        vec4 p = permute(permute(permute(
                    i.z + vec4(0.0, i1.z, i2.z, 1.0))
                  + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                  + i.x + vec4(0.0, i1.x, i2.x, 1.0));

        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;

        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);

        vec4 x = x_ * ns.x + ns.yyyy;
        vec4 y = y_ * ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);

        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);

        vec4 s0 = floor(b0) * 2.0 + 1.0;
        vec4 s1 = floor(b1) * 2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));

        vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

        vec3 g0 = vec3(a0.xy, h.x);
        vec3 g1 = vec3(a0.zw, h.y);
        vec3 g2 = vec3(a1.xy, h.z);
        vec3 g3 = vec3(a1.zw, h.w);

        vec4 norm = taylorInvSqrt(vec4(dot(g0, g0), dot(g1, g1),
                                        dot(g2, g2), dot(g3, g3)));
        g0 *= norm.x;
        g1 *= norm.y;
        g2 *= norm.z;
        g3 *= norm.w;

        vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1),
                                dot(x2, x2), dot(x3, x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m * m, vec4(dot(g0, x0), dot(g1, x1),
                                       dot(g2, x2), dot(g3, x3)));
    }

    // snoise returns ~[-1, 1]; remap to [0, 1] for drop-in compatibility
    // with the prior value-noise-based bumpField.
    float snoise01(vec3 v) {
        return snoise(v) * 0.5 + 0.5;
    }

    // High-frequency, time-varying bump field used as a fake normal map for
    // the surface micro-relief. 3D simplex noise with time on the third
    // axis: the XY sample point translates by drift (coherent flow
    // direction) while the Z coordinate advances at evolutionRate (in-place
    // morph, independent of any flow). Two octaves at independently
    // tunable scales (uBumpOctaveScales) and weights (uBumpOctaveWeights);
    // drift is scaled per-octave too so the visible flow speed stays
    // consistent across both octaves regardless of their relative
    // frequencies.
    float bumpField(vec2 p, float t, vec2 drift, float evolutionRate) {
        vec2 flow = drift * t;
        float a = snoise01(vec3(uBumpOctaveScales.x * (p + flow), t * evolutionRate));
        float b = snoise01(vec3(uBumpOctaveScales.y * (p - flow), t * evolutionRate * 0.7));
        return a * uBumpOctaveWeights.x + b * uBumpOctaveWeights.y;
    }

    // Finite-difference surface normal in plane-local space (z-up).
    vec3 bumpNormal(vec2 p, float t, vec2 drift, float evolutionRate, float eps, float strength) {
        float c = bumpField(p, t, drift, evolutionRate);
        float dx = (bumpField(p + vec2(eps, 0.0), t, drift, evolutionRate) - c) / eps;
        float dy = (bumpField(p + vec2(0.0, eps), t, drift, evolutionRate) - c) / eps;
        return normalize(vec3(-dx * strength, -dy * strength, 1.0));
    }

    void main() {
        // Flat tinted base — depth-based extinction (below) does the heavy
        // lifting for body colour variation, so the surface no longer needs
        // a separate noise-modulated body field. Fresnel-proxy tint shift
        // happens up-front so highlights/foam still punch through cleanly.
        vec3 base = applyFresnelTint(uWaterTint, vViewNormal);

        // --- Bump-driven specular highlights ---
        // World-anchored bump field: same scene fragment looks at a
        // different noise-world XZ as the slice pans, so the highlights
        // travel with the underlying world. uBumpFrequency was authored
        // in UV-space cycles across the plane (cycles per uPlaneSize
        // world units), so divide to convert to cycles per world unit.
        vec2 p = (vWorldPos.xz + uWaterOffset) * (uBumpFrequency / uPlaneSize);

        vec3 nLocal = bumpNormal(p, uTime, uBumpDrift, uBumpEvolutionRate, 0.05, uBumpStrength);
        // The water plane is rotated -π/2 about X, so local (x, y, z) maps
        // to world (x, z, -y). Take the local-tangent normal into world.
        vec3 nWorld = vec3(nLocal.x, nLocal.z, -nLocal.y);

        // View-independent highlight direction. Using a fixed direction
        // (instead of a view-dependent half-vector) keeps the highlight
        // pattern stable as the camera rotates and tweens between snapped
        // angles — otherwise highlights would crawl during each tween,
        // re-introducing the very pixel crawl the camera snap was put in
        // place to avoid. Highlights still evolve over time because the
        // bump field animates.
        vec3 sun = normalize(uSunDirection);
        float spec = dot(nWorld, sun);

        // Two-band toon threshold: a softer glow plus a hard bright core.
        // Both run on the same spec field so the bright cores always sit
        // inside a glow halo.
        float glow = step(uGlowThreshold, spec);
        float bright = step(uBrightThreshold, spec);
        base = mix(base, uHighlightColor, glow * 0.45);
        base = mix(base, uFoamColor, bright);

        // Depth-based opacity: shallow water (e.g. above a dune crest) has
        // a short path to the seabed and stays translucent; deep water
        // approaches full opacity. Foam highlights are forced opaque so
        // they punch cleanly through the underlying water tint.
        float d = waterPathLength(gl_FragCoord.z);
        float baseAlpha = extinctionAlpha(d);

        // Shore band: a thin bright rim where path length → 0, i.e. where
        // terrain breaks the surface. Independent from the foam highlights
        // (which fire on surface normal); this fires on water depth, so it
        // outlines the waterline regardless of wave direction.
        float shore = (1.0 - smoothstep(0.0, uShoreBandWidth, d)) * uShoreStrength;
        base = mix(base, uFoamColor, shore);
        baseAlpha = max(baseAlpha, shore);

        float alpha = mix(baseAlpha, 1.0, bright);
        gl_FragColor = vec4(base, alpha);
    }
`;

// Side-panel shader. No waves, no surface highlights — just the Beer–
// Lambert extinction of light passing through the water volume from the
// camera-facing side of the panel to whatever opaque surface the view ray
// hits. With path length sampled from the depth pre-pass, the panel
// naturally fades toward translucent near the seabed (short path) and
// toward opaque tint where it covers deep water.
export const waterPanelVertexShader = /* glsl */ `
    varying vec3 vViewNormal;

    void main() {
        vViewNormal = normalize(normalMatrix * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
`;

export const waterPanelFragmentShader = /* glsl */ `
    precision highp float;
    ${EXTINCTION_GLSL}

    varying vec3 vViewNormal;

    void main() {
        vec3 base = applyFresnelTint(uWaterTint, vViewNormal);
        float d = waterPathLength(gl_FragCoord.z);
        float alpha = extinctionAlpha(d);
        gl_FragColor = vec4(base, alpha);
    }
`;
