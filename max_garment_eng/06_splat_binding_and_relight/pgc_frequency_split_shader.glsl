// 06_splat_binding_and_relight/pgc_frequency_split_shader.glsl
// Clean-room PGC-inspired frequency-split relighting & sRGB Black-Cloth Guardrail prototype.
// Keeps high-frequency Gaussian yarn/weave/stitch residual while relighting low-frequency diffuse PBR
// under named light recipes (BS_STUDIO_01 / BS_NIGHT_LUSAIL), compositing strictly in sRGB (no linear bloom).

precision highp float;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform float uHighPassGain;
uniform int uActiveGarmentId; // -1 = all, >= 0 highlights id(k) == g

in vec3 vNormalWorld;
in vec3 vGaussianSrgb;
in vec3 vGaussianLowPassSrgb;
flat in int vGarmentId;

out vec4 fragColor;

vec3 linearToSrgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 L = normalize(uLightDir);
    float ndotl = max(dot(N, L), 0.0);

    // Low-frequency relightable PBR diffuse irradiance (linear -> sRGB)
    vec3 lowLinear = vGaussianLowPassSrgb * (uAmbientColor + uLightColor * ndotl);
    vec3 lowSrgb = linearToSrgb(max(lowLinear, vec3(0.0)));

    // High-frequency captured Gaussian weave/fuzz detail residual in sRGB
    vec3 highDetailSrgb = (vGaussianSrgb - vGaussianLowPassSrgb) * uHighPassGain;

    // Direct sRGB composite preserves deep obsidian blacks (no linear bloom washout)
    vec3 finalSrgb = clamp(lowSrgb + highDetailSrgb, vec3(0.0), vec3(1.0));

    if (uActiveGarmentId >= 0 && vGarmentId == uActiveGarmentId) {
        finalSrgb = mix(finalSrgb, vec3(0.95, 0.85, 0.72), 0.14);
    }

    fragColor = vec4(finalSrgb, 1.0);
}
