// 07_web_scratch_studio/spark_hybrid_viewer.js
// Unified Three.js (>=0.180) + lazy-loaded @sparkjsdev/spark@2.4.0 (MIT) WebGL2 Mesh + Splat Fusion.
// Enforces:
//   - On-demand dynamic import of @sparkjsdev/spark@2.4.0 (never bundled on initial grid load)
//   - THREE.SRGBColorSpace + THREE.NoToneMapping (tone-map inside opaque GLB shader only; zero linear bloom)
//   - Exact Blender Z-up (-Math.PI / 2 around X) to glTF Y-up co-registration
//   - Channel-agnostic 3-ch RGB / 4-ch RGBA decoding for blind garment_id:u16 WebP (R + (G << 8))

import { evaluateGarmentMorphEnvelope } from "./maximus_two_wave_morph.js";

export const SPARK_VERIFIED_API = {
  package: "@sparkjsdev/spark@2.4.0",
  threePeer: ">=0.180.0",
  exports: ["SparkRenderer", "SplatMesh", "SplatLoader", "SplatSkinning", "SplatFileType", "dyno"],
  extensions: ["ply", "spz", "sog", "rad"],
  shThrottleCos2Deg: 0.9993908,
  insetDepthProxyMm: 1.5,
  sogZUpToYUpRotX: -Math.PI / 2,
  colorSpaceGuardrail: "THREE.SRGBColorSpace + THREE.NoToneMapping; tone-map opaque GLB shader only",
};

export const BS_RINGER_001_ASSETS = {
  glbUrl: "./assets/bs_ringer_001/maximus_v2_dressed_ringer_lod1.glb",
  thetaFitUrl: "./assets/bs_ringer_001/m_theta_fit.json",
  sogUrl: "./assets/bs_ringer_001/ringer_v3.sog",
  spzUrl: "./assets/bs_ringer_001/ringer_v4.spz",
  garmentIdUrl: "./assets/bs_ringer_001/ringer_garment_id.webp",
  videoAv1Url: "./assets/bs_ringer_001/ringer_swag01_av1.mp4",
  videoH264Url: "./assets/bs_ringer_001/ringer_swag01_h264.mp4",
  posterUrl: "./assets/bs_ringer_001/ringer_swag01_poster.webp",
};

export function createSparkRendererConfig(THREE) {
  return {
    rendererOptions: { antialias: false, powerPreference: "high-performance", alpha: false },
    outputColorSpace: THREE.SRGBColorSpace,
    toneMapping: THREE.NoToneMapping,
    maxStdDev: Math.sqrt(5),
  };
}

export function createInsetDepthProxyMaterial(THREE) {
  return new THREE.MeshBasicMaterial({
    colorWrite: false,
    depthWrite: true,
    depthTest: true,
    polygonOffset: true,
    polygonOffsetFactor: 2,
    polygonOffsetUnits: 2,
  });
}

export function attachOpaqueGlbToneMap(material) {
  material.toneMapped = true;
  material.onBeforeCompile = (shader) => {
    const helperGlsl = `
      vec3 maximusNeutralToneMapping(vec3 color) {
        const float StartCompression = 0.76;
        const float Desaturation = 0.15;
        float x = min(color.r, min(color.g, color.b));
        float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
        color -= offset;
        float peak = max(color.r, max(color.g, color.b));
        if (peak < StartCompression) return color;
        float d = 1.0 - StartCompression;
        float newPeak = 1.0 - d * d / (peak + d - StartCompression);
        color *= newPeak / peak;
        float g = 1.0 - 1.0 / (Desaturation * (peak - newPeak) + 1.0);
        return mix(color, vec3(newPeak), g);
      }
      void main() {`;
    shader.fragmentShader = shader.fragmentShader
      .replace("void main() {", helperGlsl)
      .replace(
        "#include <tonemapping_fragment>",
        "gl_FragColor.rgb = maximusNeutralToneMapping(gl_FragColor.rgb);"
      );
  };
  material.needsUpdate = true;
}

export function shouldUpdateShLut(prevViewDir, currViewDir) {
  const dot =
    prevViewDir[0] * currViewDir[0] +
    prevViewDir[1] * currViewDir[1] +
    prevViewDir[2] * currViewDir[2];
  return dot < SPARK_VERIFIED_API.shThrottleCos2Deg;
}

/**
 * Decode little-endian uint16 garment IDs (R + (G << 8)) from either 3-channel RGB
 * or 4-channel RGBA packed pixel buffers (without assuming an alpha channel).
 */
export function decodeGarmentIdBuffer(pixelData, width, height, channels = 4, maxCount = width * height) {
  if (channels !== 3 && channels !== 4) {
    throw new Error(`Unsupported garment_id pixel stride: ${channels}`);
  }
  const totalPixels = Math.min(width * height, Math.floor(pixelData.length / channels));
  const count = Math.min(maxCount, totalPixels);
  const ids = new Uint16Array(count);
  const histogram = new Map();
  for (let i = 0; i < count; i++) {
    const base = i * channels;
    const r = pixelData[base];
    const g = pixelData[base + 1];
    const gid = r | (g << 8);
    ids[i] = gid;
    histogram.set(gid, (histogram.get(gid) || 0) + 1);
  }
  return { ids, count, width, height, channels, histogram };
}

export async function loadGarmentIdWebp(url, expectedSplats = 360000) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch garment ID WebP (${res.status}): ${url}`);
  const blob = await res.blob();
  const bitmap = await createImageBitmap(blob);
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const imgData = ctx.getImageData(0, 0, width, height);
  bitmap.close?.();
  return decodeGarmentIdBuffer(imgData.data, width, height, 4, expectedSplats);
}

export function buildMaximusGarmentModifier(dyno, {
  packedTarget, garmentIdTex, shLutA, shLutB, shIndexTexA, shIndexTexB, uniforms
}) {
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    const mod = new dyno.Dyno({
      inTypes: {
        gsplat: dyno.Gsplat, target: dyno.TPackedSplats,
        uMode: "int", uProgress: "float", uSwapMask: "int", uHoverId: "int",
        uWaveDir: "vec3", uWaveGap: "float", uWaveWidth: "float", uDotRadius: "float",
        uTexWidth: "int", uGarmentId: "usampler2D", uShIdxA: "usampler2D",
        uShIdxB: "usampler2D", uShLutA: "sampler2D", uShLutB: "sampler2D"
      },
      outTypes: { gsplat: dyno.Gsplat },
      statements: ({ inputs, outputs }) => dyno.unindentLines(`
        Gsplat a = ${inputs.gsplat};
        int idx = a.index;
        ivec2 uv = ivec2(idx % ${inputs.uTexWidth}, idx / ${inputs.uTexWidth});
        uint gid = texelFetch(${inputs.uGarmentId}, uv, 0).r;

        uint shA = texelFetch(${inputs.uShIdxA}, uv, 0).r;
        vec3 shRgbA = texelFetch(${inputs.uShLutA}, ivec2(int(shA & 255u), int(shA >> 8u)), 0).rgb;
        vec3 colA = clamp(a.rgba.rgb + shRgbA, 0.0, 1.0);

        bool isSwapping = ((uint(${inputs.uSwapMask}) & (1u << gid)) != 0u);
        Gsplat outG = a;
        outG.rgba.rgb = colA;

        if (isSwapping && ${inputs.uProgress} > 0.0001) {
          Gsplat b = readPackedSplat(${inputs.target}, idx);
          uint shB = texelFetch(${inputs.uShIdxB}, uv, 0).r;
          vec3 shRgbB = texelFetch(${inputs.uShLutB}, ivec2(int(shB & 255u), int(shB >> 8u)), 0).rgb;
          vec3 colB = clamp(b.rgba.rgb + shRgbB, 0.0, 1.0);
          vec4 qB = (dot(a.quaternion, b.quaternion) < 0.0) ? -b.quaternion : b.quaternion;

          if (${inputs.uMode} == 0) {
            float t = smoothstep(0.0, 1.0, ${inputs.uProgress});
            outG.center     = mix(a.center, b.center, t);
            outG.scales     = exp(mix(log(max(a.scales, vec3(1e-5))), log(max(b.scales, vec3(1e-5))), t));
            outG.quaternion = normalize(mix(a.quaternion, qB, t));
            outG.rgba       = vec4(mix(colA, colB, t), mix(a.rgba.a, b.rgba.a, t));
          } else {
            float proj = dot(a.center, ${inputs.uWaveDir});
            float front = mix(-1.2, 1.2 + ${inputs.uWaveGap}, ${inputs.uProgress});
            float w1 = smoothstep(front - ${inputs.uWaveWidth}, front + ${inputs.uWaveWidth}, proj);
            float w2 = smoothstep((front - ${inputs.uWaveGap}) - ${inputs.uWaveWidth},
                                  (front - ${inputs.uWaveGap}) + ${inputs.uWaveWidth}, proj);
            float collapsed = clamp((1.0 - w1) - (1.0 - w2), 0.0, 1.0);
            float tMove = smoothstep(0.0, 1.0, 1.0 - w1);
            float tDone = 1.0 - w2;
            outG.center     = mix(a.center, b.center, tMove);
            outG.scales     = mix(mix(a.scales, b.scales, tDone), vec3(${inputs.uDotRadius}), collapsed);
            outG.quaternion = normalize(mix(a.quaternion, qB, tDone));
            outG.rgba       = vec4(mix(colA, colB, tMove), mix(a.rgba.a, b.rgba.a, tDone));
          }
        }
        if (${inputs.uHoverId} != 65535 && gid == uint(${inputs.uHoverId})) {
          outG.rgba.rgb = mix(outG.rgba.rgb, vec3(0.95, 0.94, 0.92), 0.14);
        }
        ${outputs.gsplat} = outG;
      `)
    });
    return { gsplat: mod.apply({ gsplat, target: packedTarget.dyno, ...uniforms }).gsplat };
  });
}

export async function lazyLoadSparkModule() {
  return import("@sparkjsdev/spark");
}

function summarizeBox3(THREE, objOrBox) {
  const box = objOrBox.isBox3 ? objOrBox.clone() : new THREE.Box3().setFromObject(objOrBox, true);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  return {
    min: [Number(box.min.x.toFixed(4)), Number(box.min.y.toFixed(4)), Number(box.min.z.toFixed(4))],
    max: [Number(box.max.x.toFixed(4)), Number(box.max.y.toFixed(4)), Number(box.max.z.toFixed(4))],
    size: [Number(size.x.toFixed(4)), Number(size.y.toFixed(4)), Number(size.z.toFixed(4))],
    center: [Number(center.x.toFixed(4)), Number(center.y.toFixed(4)), Number(center.z.toFixed(4))],
  };
}

function applyThetaFitAndFloorGround(THREE, rootObj, thetaFitSpec) {
  rootObj.updateMatrixWorld(true);
  const bySanitized = {};
  for (const [k, v] of Object.entries(thetaFitSpec?.bones || {})) {
    bySanitized[k] = v;
    bySanitized[k.replaceAll(".", "")] = v;
  }
  const skinnedMeshes = [];
  rootObj.traverse((obj) => {
    if (obj.isSkinnedMesh && obj.skeleton) {
      skinnedMeshes.push(obj);
      for (const bn of obj.skeleton.bones) {
        const spec = bySanitized[bn.name];
        if (!spec) continue;
        const [qw, qx, qy, qz] = spec.q_wxyz;
        bn.quaternion.multiply(new THREE.Quaternion(qx, qy, qz, qw));
        if (spec.t) {
          bn.position.x += spec.t[0];
          bn.position.y += spec.t[1];
          bn.position.z += spec.t[2];
        }
      }
    }
  });
  rootObj.updateMatrixWorld(true);
  for (const sm of skinnedMeshes) sm.skeleton.update();

  // Compute true skinned world bounding box across all meshes and ground sole to 0.002 mm (0.000002 m)
  const worldBox = new THREE.Box3();
  const v = new THREE.Vector3();
  rootObj.traverse((obj) => {
    if (!obj.isMesh || !obj.geometry) return;
    const pos = obj.geometry.getAttribute("position");
    if (!pos) return;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      if (obj.isSkinnedMesh) {
        obj.applyBoneTransform(i, v);
      } else {
        v.applyMatrix4(obj.matrixWorld);
      }
      worldBox.expandByPoint(v);
    }
  });
  const soleLiftY = 0.000002 - worldBox.min.y;
  for (const sm of skinnedMeshes) {
    const rootBone = sm.skeleton.bones.find((bn) => bn.name === "root");
    if (rootBone) rootBone.position.y += soleLiftY;
  }
  rootObj.updateMatrixWorld(true);
  for (const sm of skinnedMeshes) sm.skeleton.update();

  worldBox.makeEmpty();
  rootObj.traverse((obj) => {
    if (!obj.isMesh || !obj.geometry) return;
    const pos = obj.geometry.getAttribute("position");
    if (!pos) return;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      if (obj.isSkinnedMesh) {
        obj.applyBoneTransform(i, v);
      } else {
        v.applyMatrix4(obj.matrixWorld);
      }
      worldBox.expandByPoint(v);
    }
  });
  return summarizeBox3(THREE, worldBox);
}

/**
 * Compute finite, high-opacity world bounds from a Spark SplatMesh after applying its Object3D transform.
 */
function summarizeSplatWorldBounds(THREE, splatMesh, alphaThresh = 0.1) {
  splatMesh.updateMatrixWorld(true);
  const box = new THREE.Box3();
  const pt = new THREE.Vector3();
  const total = splatMesh.splats?.getNumSplats?.() ?? 0;
  let kept = 0;
  splatMesh.forEachSplat((_, center, _scales, _quat, opacity) => {
    if (opacity < alphaThresh) return;
    pt.copy(center).applyMatrix4(splatMesh.matrixWorld);
    box.expandByPoint(pt);
    kept++;
  });
  return { total, kept, ...summarizeBox3(THREE, box) };
}

export async function mountSparkHybridBenchmark(container, hudEl, options = {}) {
  const assets = { ...BS_RINGER_001_ASSETS, ...(options.assets || {}) };
  const state = {
    look: 0,
    size: "L",
    garmentHighlight: -1,
    uMorph: 0,
    mode: 0,
    viewMode: options.initialViewMode || "overlay", // "glb" | "sog" | "spz" | "overlay"
    sparkLoaded: false,
    ready: false,
    loadTimesMs: {},
    bounds: {},
    alignment: {},
    garmentId: null,
    fps: 0,
    lastFrameTimeMs: 0,
    error: null,
  };

  let THREE = null;
  let Spark = null;
  let renderer = null;
  let scene = null;
  let camera = null;
  let controls = null;
  let sparkRenderer = null;
  let glbGroup = null;
  let glbWireOverlay = null;
  let sogMesh = null;
  let spzMesh = null;
  let disposed = false;
  let rafId = 0;
  let dirty = true;
  let continuousRender = false;
  let dragging = false;

  const scheduleRender = () => {
    dirty = true;
    if (!rafId && !disposed && renderer && scene && camera) {
      rafId = requestAnimationFrame(renderFrame);
    }
  };

  let fpsWindowStart = 0;
  let fpsWindowCount = 0;
  function renderFrame(now) {
    rafId = 0;
    if (disposed || !renderer || !scene || !camera) return;
    const moved = controls ? controls.update() : false;
    if (moved || dirty || continuousRender) {
      renderer.render(scene, camera);
      dirty = false;
      if (!fpsWindowStart) fpsWindowStart = now;
      fpsWindowCount++;
      const elapsed = now - fpsWindowStart;
      if (elapsed >= 400 || fpsWindowCount >= 6) {
        state.fps = elapsed > 0 ? (fpsWindowCount * 1000) / elapsed : 0;
        fpsWindowStart = now;
        fpsWindowCount = 0;
        formatHud();
      }
      state.lastFrameTimeMs = now;
    }
    if (moved || continuousRender || dragging) {
      rafId = requestAnimationFrame(renderFrame);
    }
  }

  const formatHud = () => {
    const env = evaluateGarmentMorphEnvelope(
      state.uMorph,
      state.garmentHighlight >= 0 ? state.garmentHighlight : 1,
      new Set([1, 2])
    );
    const gidText = state.garmentId
      ? `${state.garmentId.count} texels (${state.garmentId.width}x${state.garmentId.height}), id=1:${state.garmentId.histogram.get(1) || 0}`
      : "loading...";
    const alignText = state.alignment.glbVsSogFloorDiffMm !== undefined
      ? `rotX=${state.alignment.sogRotXRad.toFixed(4)} rad (-pi/2) | floorDiff=${state.alignment.glbVsSogFloorDiffMm.toFixed(3)} mm | spzFloorDiff=${(state.alignment.glbVsSpzFloorDiffMm ?? 0).toFixed(3)} mm`
      : "measuring bounds...";
    const loadText = Object.entries(state.loadTimesMs)
      .map(([k, v]) => `${k}=${v}ms`)
      .join(" ");
    hudEl.textContent = [
      `Runtime: Three.js WebGL2 + ${SPARK_VERIFIED_API.package} | Mode: ${state.viewMode.toUpperCase()} | Ready: ${state.ready}`,
      `Color: ${SPARK_VERIFIED_API.colorSpaceGuardrail}`,
      `Load (ms): ${loadText || "pending"} | FPS: ${state.fps.toFixed(1)}`,
      `Alignment (Z-up -> Y-up): ${alignText}`,
      `Garment ID WebP: ${gidText} | Look: ${state.look} | Size: ${state.size} | Hover ID: ${state.garmentHighlight}`,
      `Two-Wave Morph Probe: scale=${env.scale.toFixed(3)} moveT=${env.moveT.toFixed(3)}`,
      state.error ? `Error: ${state.error}` : "Status: OK (zero fallback stubs)",
    ].join("\n");
    container.dataset.ready = state.ready ? "1" : "0";
    container.dataset.viewMode = state.viewMode;
    container.dataset.fps = state.fps.toFixed(1);
  };

  const applyViewMode = () => {
    if (glbGroup) glbGroup.visible = state.viewMode === "glb";
    if (glbWireOverlay) glbWireOverlay.visible = state.viewMode === "overlay";
    if (sogMesh) sogMesh.visible = state.viewMode === "sog" || state.viewMode === "overlay";
    if (spzMesh) spzMesh.visible = state.viewMode === "spz";
    if (sogMesh) {
      const tint = state.garmentHighlight === 1 ? 0xffe4b8 : 0xffffff;
      sogMesh.recolor.setHex(tint);
    }
    if (spzMesh) {
      const tint = state.garmentHighlight === 1 ? 0xffe4b8 : 0xffffff;
      spzMesh.recolor.setHex(tint);
    }
    formatHud();
  };

  const initScene = async () => {
    const tTotalStart = performance.now();
    const [threeMod, gltfMod, orbitMod, envMod, meshoptMod, sparkMod] = await Promise.all([
      import("three"),
      import("three/addons/loaders/GLTFLoader.js"),
      import("three/addons/controls/OrbitControls.js"),
      import("three/addons/environments/RoomEnvironment.js"),
      import("meshoptimizer/decoder"),
      lazyLoadSparkModule(),
    ]);
    THREE = threeMod;
    Spark = sparkMod;
    state.sparkLoaded = true;

    const cfg = createSparkRendererConfig(THREE);
    renderer = new THREE.WebGLRenderer(cfg.rendererOptions);
    renderer.outputColorSpace = cfg.outputColorSpace;
    renderer.toneMapping = cfg.toneMapping;
    const glCtx = renderer.getContext();
    const dbgExt = glCtx.getExtension("WEBGL_debug_renderer_info");
    const glRendererName = dbgExt ? String(glCtx.getParameter(dbgExt.UNMASKED_RENDERER_WEBGL) || "") : "";
    const isSoftwareGl = /swiftshader|llvmpipe|softpipe/i.test(glRendererName);
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x0a0a0c, 1);

    const w = container.clientWidth || 540;
    const h = container.clientHeight || 750;
    renderer.setSize(w, h, false);
    const canvas = renderer.domElement;
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    container.innerHTML = "";
    container.appendChild(canvas);

    scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new envMod.RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.45;
    const key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(1.8, 2.4, 2.4);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xeef2ff, 0.45);
    fill.position.set(-2.2, 1.6, 1.8);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xfff4e6, 0.55);
    rim.position.set(0.2, 2.0, -2.4);
    scene.add(rim);

    // Camera positioned on +Z looking toward torso/full body center (y = 0.88m)
    camera = new THREE.PerspectiveCamera(28, w / h, 0.05, 50);
    camera.position.set(0, 0.94, 4.15);
    controls = new orbitMod.OrbitControls(camera, canvas);
    controls.target.set(0, 0.88, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();
    controls.addEventListener("start", () => { dragging = true; scheduleRender(); });
    controls.addEventListener("end", () => { dragging = false; scheduleRender(); });
    controls.addEventListener("change", scheduleRender);

    sparkRenderer = new Spark.SparkRenderer({
      renderer,
      maxStdDev: cfg.maxStdDev,
      onDirty: scheduleRender,
    });
    scene.add(sparkRenderer);

    // Floor reference ring (y = 0) to confirm 0.002 mm foot grounding visually
    const grid = new THREE.GridHelper(2.0, 10, 0x3b3832, 0x1d1c20);
    grid.position.y = 0;
    scene.add(grid);

    // 1. Load GLB (Meshopt + KHR_mesh_quantization) + pose SkinnedMesh into theta_fit
    const tGlbStart = performance.now();
    await meshoptMod.MeshoptDecoder.ready;
    const gltfLoader = new gltfMod.GLTFLoader();
    gltfLoader.setMeshoptDecoder(meshoptMod.MeshoptDecoder);
    const [gltf, wireGltf, thetaFitSpec] = await Promise.all([
      gltfLoader.loadAsync(assets.glbUrl),
      gltfLoader.loadAsync(assets.glbUrl),
      fetch(assets.thetaFitUrl).then((r) => (r.ok ? r.json() : null)),
    ]);
    glbGroup = gltf.scene;
    glbGroup.traverse((obj) => {
      if (!obj.isMesh) return;
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => {
        if (m.name === "MX_M1_ObsidianNoir") {
          // Restore M1 Obsidian Noir base albedo (~#18181B linear 0.009, 0.009, 0.011) if Blender node graph omitted scalar factor
          m.color.setRGB(0.011, 0.011, 0.013);
          m.roughness = 0.52;
        } else if (m.name === "Main_Body") {
          // COLOR_0 stores (_TENSION, collar_mask) diagnostic channels, not knit albedo
          m.vertexColors = false;
        }
        attachOpaqueGlbToneMap(m);
      });
    });
    const posedGlbBounds = applyThetaFitAndFloorGround(THREE, glbGroup, thetaFitSpec);

    glbWireOverlay = wireGltf.scene;
    glbWireOverlay.traverse((obj) => {
      if (obj.isMesh) {
        obj.material = new THREE.MeshBasicMaterial({
          color: 0x38d9a9,
          wireframe: true,
          transparent: true,
          opacity: 0.35,
        });
      }
    });
    applyThetaFitAndFloorGround(THREE, glbWireOverlay, thetaFitSpec);
    scene.add(glbGroup);
    scene.add(glbWireOverlay);
    state.loadTimesMs.glb = Math.round(performance.now() - tGlbStart);
    state.bounds.glb = posedGlbBounds;

    // 2. Load SOG v3 (Meta v2 + SH0, exported directly from Blender Z-up)
    const tSogStart = performance.now();
    sogMesh = new Spark.SplatMesh({ url: assets.sogUrl });
    await sogMesh.initialized;
    // Measure raw SOG bounds before alignment, then apply exact Blender Z-up -> glTF Y-up (-PI/2 around X)
    state.bounds.sogRaw = summarizeSplatWorldBounds(THREE, sogMesh, 0.1);
    sogMesh.rotation.x = SPARK_VERIFIED_API.sogZUpToYUpRotX;
    scene.add(sogMesh);
    state.bounds.sogAligned = summarizeSplatWorldBounds(THREE, sogMesh, 0.1);
    state.loadTimesMs.sog = Math.round(performance.now() - tSogStart);

    // 3. Load SPZ v4 (NGSP v4, exported from the same Z-up PLY)
    const tSpzStart = performance.now();
    spzMesh = new Spark.SplatMesh({ url: assets.spzUrl });
    await spzMesh.initialized;
    state.bounds.spzRaw = summarizeSplatWorldBounds(THREE, spzMesh, 0.1);
    spzMesh.rotation.x = SPARK_VERIFIED_API.sogZUpToYUpRotX;
    scene.add(spzMesh);
    state.bounds.spzAligned = summarizeSplatWorldBounds(THREE, spzMesh, 0.1);
    state.loadTimesMs.spz = Math.round(performance.now() - tSpzStart);

    // Check alignment delta between GLB Y-up and rotated SOG/SPZ
    state.alignment = {
      sogRotXRad: SPARK_VERIFIED_API.sogZUpToYUpRotX,
      spzRotXRad: SPARK_VERIFIED_API.sogZUpToYUpRotX,
      sogRawHeightZ: Number((state.bounds.sogRaw.max[2] - state.bounds.sogRaw.min[2]).toFixed(4)),
      spzRawHeightZ: Number((state.bounds.spzRaw.max[2] - state.bounds.spzRaw.min[2]).toFixed(4)),
      glbHeightY: state.bounds.glb.size[1],
      sogAlignedHeightY: state.bounds.sogAligned.size[1],
      spzAlignedHeightY: state.bounds.spzAligned.size[1],
      glbVsSogFloorDiffMm: (state.bounds.sogAligned.min[1] - state.bounds.glb.min[1]) * 1000.0,
      glbVsSpzFloorDiffMm: (state.bounds.spzAligned.min[1] - state.bounds.glb.min[1]) * 1000.0,
      glbCenterY: state.bounds.glb.center[1],
      sogCenterY: state.bounds.sogAligned.center[1],
      spzCenterY: state.bounds.spzAligned.center[1],
    };

    // 4. Load & decode 600x600 garment_id:u16 lossless WebP (handles 3-ch RGB & 4-ch RGBA safely)
    const tGidStart = performance.now();
    state.garmentId = await loadGarmentIdWebp(assets.garmentIdUrl, state.bounds.sogAligned.total || 360000);
    state.loadTimesMs.garmentId = Math.round(performance.now() - tGidStart);
    state.loadTimesMs.total = Math.round(performance.now() - tTotalStart);

    state.ready = true;
    applyViewMode();

    // Render one immediate settled frame, then run on-demand via controls/onDirty
    const tFirstRender = performance.now();
    renderer.render(scene, camera);
    const firstRenderMs = Math.max(performance.now() - tFirstRender, 1);
    state.loadTimesMs.firstRender = Math.round(firstRenderMs);
    state.fps = Number((1000 / firstRenderMs).toFixed(2));
    formatHud();
    scheduleRender();
  };

  formatHud();
  const readyPromise = initScene().catch((err) => {
    state.error = err?.message || String(err);
    formatHud();
    throw err;
  });

  const controller = {
    api: SPARK_VERIFIED_API,
    readyPromise,
    state,
    attachSparkRuntime: lazyLoadSparkModule,
    setViewMode(mode) {
      if (!["glb", "sog", "spz", "overlay"].includes(mode)) {
        throw new Error(`Unknown viewMode: ${mode}`);
      }
      state.viewMode = mode;
      applyViewMode();
      if (renderer && scene && camera) {
        renderer.render(scene, camera);
        scheduleRender();
      }
    },
    async measureFps(frameCount = 5) {
      if (!renderer || !scene || !camera) return { fps: 0, frameMs: 0 };
      const t0 = performance.now();
      for (let i = 0; i < frameCount; i++) {
        camera.position.x = 0.001 * Math.sin(i);
        controls.update();
        renderer.render(scene, camera);
        await new Promise((r) => requestAnimationFrame(r));
      }
      camera.position.x = 0;
      controls.update();
      renderer.render(scene, camera);
      const dt = Math.max(performance.now() - t0, 1);
      const fps = (frameCount * 1000) / dt;
      state.fps = fps;
      formatHud();
      return { fps: Number(fps.toFixed(2)), frameMs: Number((dt / frameCount).toFixed(1)) };
    },
    setCameraView(pos, target) {
      if (!camera || !controls) return;
      camera.position.set(pos[0], pos[1], pos[2]);
      if (target) controls.target.set(target[0], target[1], target[2]);
      controls.update();
      renderer.render(scene, camera);
    },
    renderNow() {
      if (renderer && scene && camera) renderer.render(scene, camera);
    },
    setLook(v) {
      state.look = v;
      state.uMorph = v === 0 ? 0 : 0.5;
      formatHud();
    },
    setMode(m) {
      state.mode = m;
      formatHud();
    },
    setSize(s) {
      state.size = s;
      formatHud();
    },
    setGarmentHighlight(gid) {
      state.garmentHighlight = gid;
      applyViewMode();
      if (renderer && scene && camera) renderer.render(scene, camera);
    },
    getDiagnostics() {
      return {
        ready: state.ready,
        viewMode: state.viewMode,
        loadTimesMs: state.loadTimesMs,
        fps: Number(state.fps.toFixed(2)),
        bounds: state.bounds,
        alignment: state.alignment,
        garmentId: state.garmentId
          ? {
              count: state.garmentId.count,
              width: state.garmentId.width,
              height: state.garmentId.height,
              channels: state.garmentId.channels,
              histogram: Object.fromEntries(state.garmentId.histogram.entries()),
            }
          : null,
        error: state.error,
      };
    },
    dispose() {
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      sogMesh?.dispose?.();
      spzMesh?.dispose?.();
      sparkRenderer?.dispose?.();
      controls?.dispose?.();
      renderer?.dispose?.();
    },
  };

  if (typeof window !== "undefined") {
    window.__SPARK_SCRATCH_CTRL__ = controller;
  }
  await readyPromise;
  return controller;
}
