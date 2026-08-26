import { OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import {
  ArrowDown,
  Expand,
  Focus,
  Grid3X3,
  Image as ImageIcon,
  Layers,
  Loader,
  Minimize,
  Palette,
  Play,
  RefreshCw,
  RotateCcw,
  Sparkles,
  Wand2,
} from "lucide-react";
import { type ElementRef, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { formatNumber } from "../lib/format";
import type { DepthGrid, GeospatialMetadata } from "../types/api";

const MAX_MESH_DIMENSION = 256;
const PERSPECTIVE_CAMERA_POSITION: [number, number, number] = [4.8, 4.2, 5.4];
const TOPDOWN_CAMERA_POSITION: [number, number, number] = [0, 7.8, 0];
const HEIGHT_SCALE = 0.55;

type TextureState = "idle" | "loading" | "loaded" | "failed";
type CameraPreset = "perspective" | "topdown";
type SmoothingLevel = "none" | "subtle" | "balanced" | "smooth";

interface TerrainViewerProps {
  depthGrid: DepthGrid;
  textureUrl: string | null;
  geospatial: GeospatialMetadata | null;
  inputWidth: number;
  inputHeight: number;
  valueLabel: string;
}

interface PointSample {
  pixelX: number;
  pixelY: number;
  value: number;
  mapX?: number;
  mapY?: number;
  coordinateKind?: "geographic" | "projected";
}

interface TerrainData {
  geometry: THREE.BufferGeometry;
  columns: number;
  rows: number;
  minimum: number;
  maximum: number;
}

function getGridDimensions(grid: DepthGrid): { width: number; height: number } {
  if (Array.isArray(grid.values[0])) {
    const rows = grid.values as number[][];
    return {
      width: Math.max(1, Math.min(grid.width || rows[0]?.length || 1, rows[0]?.length || 1)),
      height: Math.max(1, Math.min(grid.height || rows.length, rows.length)),
    };
  }

  const values = grid.values as number[];
  const width = Math.max(1, grid.width || Math.round(Math.sqrt(values.length)) || 1);
  const height = Math.max(1, Math.min(grid.height || Math.ceil(values.length / width), Math.ceil(values.length / width)));
  return { width, height };
}

function getGridValue(grid: DepthGrid, x: number, y: number): number {
  if (Array.isArray(grid.values[0])) {
    const rows = grid.values as number[][];
    return Number(rows[y]?.[x]);
  }
  return Number((grid.values as number[])[y * grid.width + x]);
}

function isGridValueValid(grid: DepthGrid, x: number, y: number): boolean {
  const mask = grid.valid_mask;
  if (!mask) return true;
  if (Array.isArray(mask[0])) return Boolean((mask as boolean[][])[y]?.[x]);
  return Boolean((mask as boolean[])[y * grid.width + x]);
}

/** Bilinear subpixel sampling for smooth continuous height interpolation (no terracing). */
function sampleBilinear(grid: DepthGrid, u: number, v: number, width: number, height: number): number {
  const gx = u * (width - 1);
  const gy = v * (height - 1);
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(gx)));
  const x1 = Math.max(0, Math.min(width - 1, Math.ceil(gx)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(gy)));
  const y1 = Math.max(0, Math.min(height - 1, Math.ceil(gy)));
  const fx = gx - x0;
  const fy = gy - y0;

  const v00 = getGridValue(grid, x0, y0);
  const v10 = getGridValue(grid, x1, y0);
  const v01 = getGridValue(grid, x0, y1);
  const v11 = getGridValue(grid, x1, y1);

  const top = v00 * (1 - fx) + v10 * fx;
  const bottom = v01 * (1 - fx) + v11 * fx;
  return top * (1 - fy) + bottom * fy;
}

/** Edge-preserving 3x3 Gaussian smoothing filter to remove sensor noise and terracing artifacts. */
function applySmoothing(samples: Float32Array, cols: number, rows: number, passes: number): void {
  if (passes <= 0) return;

  const temp = new Float32Array(samples.length);

  for (let p = 0; p < passes; p += 1) {
    temp.set(samples);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const idx = r * cols + c;
        if (r === 0 || r === rows - 1 || c === 0 || c === cols - 1) {
          continue;
        }

        const center = temp[idx];
        const n = temp[(r - 1) * cols + c];
        const s = temp[(r + 1) * cols + c];
        const w = temp[r * cols + (c - 1)];
        const e = temp[r * cols + (c + 1)];
        const nw = temp[(r - 1) * cols + (c - 1)];
        const ne = temp[(r - 1) * cols + (c + 1)];
        const sw = temp[(r + 1) * cols + (c - 1)];
        const se = temp[(r + 1) * cols + (c + 1)];

        samples[idx] = (center * 4 + (n + s + w + e) * 2 + (nw + ne + sw + se) * 1) / 16;
      }
    }
  }
}

/** Optional local relief detrending (removes dominant perspective tilt from monocular models). */
function applyDetrend(samples: Float32Array, cols: number, rows: number): void {
  let sumX = 0, sumY = 0, sumZ = 0, sumXX = 0, sumYY = 0, sumXY = 0, sumXZ = 0, sumYZ = 0;
  const n = cols * rows;

  for (let r = 0; r < rows; r += 1) {
    const y = r / (rows - 1) - 0.5;
    for (let c = 0; c < cols; c += 1) {
      const x = c / (cols - 1) - 0.5;
      const z = samples[r * cols + c];
      sumX += x; sumY += y; sumZ += z;
      sumXX += x * x; sumYY += y * y; sumXY += x * y;
      sumXZ += x * z; sumYZ += y * z;
    }
  }

  const denomX = sumXX || 1;
  const denomY = sumYY || 1;
  const slopeX = (sumXZ - (sumX * sumZ) / n) / denomX;
  const slopeY = (sumYZ - (sumY * sumZ) / n) / denomY;

  for (let r = 0; r < rows; r += 1) {
    const y = r / (rows - 1) - 0.5;
    for (let c = 0; c < cols; c += 1) {
      const x = c / (cols - 1) - 0.5;
      const idx = r * cols + c;
      const trend = slopeX * x + slopeY * y;
      samples[idx] = samples[idx] - trend * 0.7;
    }
  }
}

function terrainColor(normalized: number): THREE.Color {
  const stops = [
    { at: 0.0, color: new THREE.Color("#0c2033") },
    { at: 0.22, color: new THREE.Color("#15536e") },
    { at: 0.48, color: new THREE.Color("#2a8c88") },
    { at: 0.74, color: new THREE.Color("#6ad19f") },
    { at: 1.0, color: new THREE.Color("#fae28c") },
  ];
  const upperIndex = stops.findIndex((stop) => normalized <= stop.at);
  if (upperIndex <= 0) return stops[0].color.clone();
  const lower = stops[upperIndex - 1];
  const upper = stops[upperIndex];
  const amount = (normalized - lower.at) / (upper.at - lower.at || 1);
  return lower.color.clone().lerp(upper.color, amount);
}

function createTerrainGeometry(
  grid: DepthGrid,
  smoothing: SmoothingLevel,
  detrend: boolean,
  withSkirt: boolean,
): TerrainData {
  const dimensions = getGridDimensions(grid);
  const columns = Math.min(MAX_MESH_DIMENSION, dimensions.width);
  const rows = Math.min(MAX_MESH_DIMENSION, dimensions.height);
  const samples = new Float32Array(columns * rows);
  const validSamples = new Uint8Array(columns * rows);
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;

  for (let row = 0; row < rows; row += 1) {
    const v = row / Math.max(1, rows - 1);
    for (let column = 0; column < columns; column += 1) {
      const u = column / Math.max(1, columns - 1);
      const value = sampleBilinear(grid, u, v, dimensions.width, dimensions.height);
      const index = row * columns + column;
      const sourceX = Math.round(u * (dimensions.width - 1));
      const sourceY = Math.round(v * (dimensions.height - 1));
      const valid = Number.isFinite(value) && isGridValueValid(grid, sourceX, sourceY);
      samples[index] = Number.isFinite(value) ? value : 0;
      validSamples[index] = valid ? 1 : 0;
      if (valid) {
        minimum = Math.min(minimum, value);
        maximum = Math.max(maximum, value);
      }
    }
  }

  if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) {
    minimum = 0;
    maximum = 1;
  }

  // Apply smoothing
  const passes = smoothing === "none" ? 0 : smoothing === "subtle" ? 1 : smoothing === "balanced" ? 2 : 4;
  if (passes > 0) {
    applySmoothing(samples, columns, rows, passes);
  }

  // Optional detrending
  if (detrend) {
    applyDetrend(samples, columns, rows);
  }

  const range = maximum - minimum || 1;
  const alreadyNormalized = minimum >= -0.01 && maximum <= 1.01 && range <= 1.02;
  const sourceAspect = dimensions.width / Math.max(1, dimensions.height);
  const width = sourceAspect >= 1 ? 5.6 : 5.6 * sourceAspect;
  const depth = sourceAspect >= 1 ? 5.6 / sourceAspect : 5.6;

  const topVertexCount = columns * rows;
  const skirtPerimeterCount = columns * 2 + (rows - 2) * 2;
  const totalVertices = withSkirt ? topVertexCount + skirtPerimeterCount * 4 + 4 : topVertexCount;

  const positions = new Float32Array(totalVertices * 3);
  const colors = new Float32Array(totalVertices * 3);
  const uvs = new Float32Array(totalVertices * 2);
  const indices: number[] = [];

  const baseDepthY = -0.55 * HEIGHT_SCALE - 0.25;

  // 1. Top Surface
  for (let row = 0; row < rows; row += 1) {
    const v = row / Math.max(1, rows - 1);
    for (let column = 0; column < columns; column += 1) {
      const u = column / Math.max(1, columns - 1);
      const index = row * columns + column;
      const normalized = alreadyNormalized
        ? Math.max(0, Math.min(1, samples[index]))
        : Math.max(0, Math.min(1, (samples[index] - minimum) / range));
      const color = terrainColor(normalized);

      positions[index * 3] = (u - 0.5) * width;
      positions[index * 3 + 1] = (normalized - 0.5) * HEIGHT_SCALE;
      positions[index * 3 + 2] = (v - 0.5) * depth;

      colors[index * 3] = color.r;
      colors[index * 3 + 1] = color.g;
      colors[index * 3 + 2] = color.b;

      uvs[index * 2] = u;
      uvs[index * 2 + 1] = 1 - v;
    }
  }

  // Top Triangles
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const topLeft = row * columns + column;
      const topRight = topLeft + 1;
      const bottomLeft = (row + 1) * columns + column;
      const bottomRight = bottomLeft + 1;
      if (validSamples[topLeft] && validSamples[bottomLeft] && validSamples[topRight]) {
        indices.push(topLeft, bottomLeft, topRight);
      }
      if (validSamples[topRight] && validSamples[bottomLeft] && validSamples[bottomRight]) {
        indices.push(topRight, bottomLeft, bottomRight);
      }
    }
  }

  // 2. Solid 3D Base Skirt
  if (withSkirt) {
    let skirtVIdx = topVertexCount;
    const skirtColor = new THREE.Color("#0c1b26");

    const perimeterIndices: number[] = [];
    for (let c = 0; c < columns; c += 1) perimeterIndices.push(0 * columns + c);
    for (let r = 1; r < rows; r += 1) perimeterIndices.push(r * columns + (columns - 1));
    for (let c = columns - 2; c >= 0; c -= 1) perimeterIndices.push((rows - 1) * columns + c);
    for (let r = rows - 2; r > 0; r -= 1) perimeterIndices.push(r * columns + 0);

    const perimeterLen = perimeterIndices.length;

    for (let i = 0; i < perimeterLen; i += 1) {
      const topIdx = perimeterIndices[i];
      const nextTopIdx = perimeterIndices[(i + 1) % perimeterLen];

      const topX = positions[topIdx * 3];
      const topY = positions[topIdx * 3 + 1];
      const topZ = positions[topIdx * 3 + 2];

      const nextTopX = positions[nextTopIdx * 3];
      const nextTopY = positions[nextTopIdx * 3 + 1];
      const nextTopZ = positions[nextTopIdx * 3 + 2];

      const vTopA = skirtVIdx++;
      const vBotA = skirtVIdx++;
      const vTopB = skirtVIdx++;
      const vBotB = skirtVIdx++;

      positions[vTopA * 3] = topX; positions[vTopA * 3 + 1] = topY; positions[vTopA * 3 + 2] = topZ;
      positions[vBotA * 3] = topX; positions[vBotA * 3 + 1] = baseDepthY; positions[vBotA * 3 + 2] = topZ;
      positions[vTopB * 3] = nextTopX; positions[vTopB * 3 + 1] = nextTopY; positions[vTopB * 3 + 2] = nextTopZ;
      positions[vBotB * 3] = nextTopX; positions[vBotB * 3 + 1] = baseDepthY; positions[vBotB * 3 + 2] = nextTopZ;

      for (const v of [vTopA, vBotA, vTopB, vBotB]) {
        colors[v * 3] = skirtColor.r;
        colors[v * 3 + 1] = skirtColor.g;
        colors[v * 3 + 2] = skirtColor.b;
        uvs[v * 2] = 0;
        uvs[v * 2 + 1] = 0;
      }

      indices.push(vTopA, vBotA, vTopB);
      indices.push(vTopB, vBotA, vBotB);
    }

    // Bottom Cap
    const b0 = skirtVIdx++;
    const b1 = skirtVIdx++;
    const b2 = skirtVIdx++;
    const b3 = skirtVIdx++;

    positions[b0 * 3] = -width / 2; positions[b0 * 3 + 1] = baseDepthY; positions[b0 * 3 + 2] = -depth / 2;
    positions[b1 * 3] = width / 2;  positions[b1 * 3 + 1] = baseDepthY; positions[b1 * 3 + 2] = -depth / 2;
    positions[b2 * 3] = width / 2;  positions[b2 * 3 + 1] = baseDepthY; positions[b2 * 3 + 2] = depth / 2;
    positions[b3 * 3] = -width / 2; positions[b3 * 3 + 1] = baseDepthY; positions[b3 * 3 + 2] = depth / 2;

    for (const v of [b0, b1, b2, b3]) {
      colors[v * 3] = 0.05; colors[v * 3 + 1] = 0.09; colors[v * 3 + 2] = 0.12;
      uvs[v * 2] = 0; uvs[v * 2 + 1] = 0;
    }

    indices.push(b0, b2, b1);
    indices.push(b0, b3, b2);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  return { geometry, columns, rows, minimum, maximum };
}

function extractBounds(geospatial: GeospatialMetadata): [number, number, number, number] | null {
  const { bounds } = geospatial;
  if (Array.isArray(bounds) && bounds.length >= 4 && bounds.slice(0, 4).every((value) => typeof value === "number")) {
    return [Number(bounds[0]), Number(bounds[1]), Number(bounds[2]), Number(bounds[3])];
  }
  if (bounds && !Array.isArray(bounds)) {
    const left = bounds.left ?? bounds.min_x;
    const bottom = bounds.bottom ?? bounds.min_y;
    const right = bounds.right ?? bounds.max_x;
    const top = bounds.top ?? bounds.max_y;
    if ([left, bottom, right, top].every((value) => typeof value === "number")) {
      return [Number(left), Number(bottom), Number(right), Number(top)];
    }
  }
  return null;
}

function projectPixel(
  pixelX: number,
  pixelY: number,
  inputWidth: number,
  inputHeight: number,
  geospatial: GeospatialMetadata | null,
): Pick<PointSample, "mapX" | "mapY" | "coordinateKind"> {
  if (!geospatial || !geospatial.crs || geospatial.valid_for_dsm_export !== true) return {};
  const transform = geospatial.transform;
  let mapX: number | undefined;
  let mapY: number | undefined;

  if (Array.isArray(transform) && transform.length >= 6 && transform.slice(0, 6).every(Number.isFinite)) {
    mapX = transform[0] * pixelX + transform[1] * pixelY + transform[2];
    mapY = transform[3] * pixelX + transform[4] * pixelY + transform[5];
  } else {
    const bounds = extractBounds(geospatial);
    if (bounds) {
      const [left, bottom, right, top] = bounds;
      mapX = left + (pixelX / Math.max(1, inputWidth - 1)) * (right - left);
      mapY = top + (pixelY / Math.max(1, inputHeight - 1)) * (bottom - top);
    }
  }

  if (mapX === undefined || mapY === undefined) return {};
  const crs = String(geospatial.crs ?? geospatial.epsg ?? "").toLowerCase();
  return {
    mapX,
    mapY,
    coordinateKind: crs.includes("4326") || crs.includes("wgs 84") ? "geographic" : "projected",
  };
}

function CameraControls({
  resetKey,
  preset,
  autoRotate,
}: {
  resetKey: number;
  preset: CameraPreset;
  autoRotate: boolean;
}) {
  const controlsRef = useRef<ElementRef<typeof OrbitControls> | null>(null);
  const { camera } = useThree();

  useEffect(() => {
    if (preset === "topdown") {
      camera.position.set(...TOPDOWN_CAMERA_POSITION);
      camera.lookAt(0, 0, 0);
      controlsRef.current?.target.set(0, 0, 0);
      controlsRef.current?.update();
    } else {
      camera.position.set(...PERSPECTIVE_CAMERA_POSITION);
      camera.lookAt(0, 0, 0);
      controlsRef.current?.target.set(0, 0, 0);
      controlsRef.current?.update();
    }
  }, [camera, resetKey, preset]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.06}
      minDistance={2.4}
      maxDistance={14}
      maxPolarAngle={Math.PI / 2.05}
      autoRotate={autoRotate}
      autoRotateSpeed={0.85}
    />
  );
}

function TerrainMesh({
  data,
  grid,
  inputWidth,
  inputHeight,
  geospatial,
  textureUrl,
  showTexture,
  wireframe,
  exaggeration,
  onPointSelected,
  onTextureStateChange,
  textureLoadKey,
}: {
  data: TerrainData;
  grid: DepthGrid;
  inputWidth: number;
  inputHeight: number;
  geospatial: GeospatialMetadata | null;
  textureUrl: string | null;
  showTexture: boolean;
  wireframe: boolean;
  exaggeration: number;
  onPointSelected: (sample: PointSample) => void;
  onTextureStateChange: (state: TextureState) => void;
  textureLoadKey: number;
}) {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    setTexture(null);
    if (!textureUrl) {
      onTextureStateChange("idle");
      return;
    }

    onTextureStateChange("loading");
    let active = true;
    let loadedTexture: THREE.Texture | null = null;
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin("anonymous");
    loader.load(
      textureUrl,
      (nextTexture) => {
        loadedTexture = nextTexture;
        nextTexture.colorSpace = THREE.SRGBColorSpace;
        nextTexture.generateMipmaps = true;
        nextTexture.minFilter = THREE.LinearMipmapLinearFilter;
        nextTexture.magFilter = THREE.LinearFilter;
        nextTexture.anisotropy = 16;
        nextTexture.wrapS = THREE.ClampToEdgeWrapping;
        nextTexture.wrapT = THREE.ClampToEdgeWrapping;
        nextTexture.needsUpdate = true;
        if (active) {
          setTexture(nextTexture);
          onTextureStateChange("loaded");
        } else {
          nextTexture.dispose();
        }
      },
      undefined,
      () => {
        if (active) {
          setTexture(null);
          onTextureStateChange("failed");
        }
      },
    );

    return () => {
      active = false;
      loadedTexture?.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [textureUrl, textureLoadKey]);

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    if (!event.uv) return;
    event.stopPropagation();
    const dimensions = getGridDimensions(grid);
    const gridX = Math.max(0, Math.min(dimensions.width - 1, Math.round(event.uv.x * (dimensions.width - 1))));
    const gridY = Math.max(0, Math.min(dimensions.height - 1, Math.round((1 - event.uv.y) * (dimensions.height - 1))));
    const pixelX = Math.round((gridX / Math.max(1, dimensions.width - 1)) * Math.max(0, inputWidth - 1));
    const pixelY = Math.round((gridY / Math.max(1, dimensions.height - 1)) * Math.max(0, inputHeight - 1));
    const value = getGridValue(grid, gridX, gridY);
    onPointSelected({
      pixelX,
      pixelY,
      value,
      ...projectPixel(pixelX, pixelY, inputWidth, inputHeight, geospatial),
    });
  };

  const hasTexture = showTexture && texture !== null;

  return (
    <mesh
      geometry={data.geometry}
      scale={[1, exaggeration, 1]}
      onClick={handleClick}
      receiveShadow
      castShadow
    >
      <meshStandardMaterial
        map={hasTexture ? texture : null}
        vertexColors={!hasTexture}
        wireframe={wireframe}
        roughness={0.82}
        metalness={0.06}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

export default function TerrainViewer({
  depthGrid,
  textureUrl,
  geospatial,
  inputWidth,
  inputHeight,
  valueLabel,
}: TerrainViewerProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [materialMode, setMaterialMode] = useState<"texture" | "height">(textureUrl ? "texture" : "height");
  const [smoothing, setSmoothing] = useState<SmoothingLevel>("balanced");
  const [detrend, setDetrend] = useState(false);
  const [withSkirt, setWithSkirt] = useState(true);
  const [wireframe, setWireframe] = useState(false);
  const [exaggeration, setExaggeration] = useState(0.4);
  const [resetKey, setResetKey] = useState(0);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("perspective");
  const [autoRotate, setAutoRotate] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sample, setSample] = useState<PointSample | null>(null);
  const [textureState, setTextureState] = useState<TextureState>("idle");
  const [textureLoadKey, setTextureLoadKey] = useState(0);

  const terrain = useMemo(
    () => createTerrainGeometry(depthGrid, smoothing, detrend, withSkirt),
    [depthGrid, smoothing, detrend, withSkirt],
  );

  useEffect(() => () => terrain.geometry.dispose(), [terrain]);

  useEffect(() => {
    const handleFullscreenChange = () => setIsFullscreen(document.fullscreenElement === wrapperRef.current);
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    setMaterialMode(textureUrl ? "texture" : "height");
    setSample(null);
    setTextureState(textureUrl ? "loading" : "idle");
  }, [textureUrl, depthGrid]);

  const toggleFullscreen = async () => {
    if (!wrapperRef.current) return;
    if (document.fullscreenElement) await document.exitFullscreen();
    else await wrapperRef.current.requestFullscreen();
  };

  const handleRetryTexture = useCallback(() => {
    setTextureLoadKey((key) => key + 1);
    setTextureState("loading");
  }, []);

  const handleTopDown = useCallback(() => {
    setCameraPreset((current) => (current === "topdown" ? "perspective" : "topdown"));
  }, []);

  const cycleSmoothing = () => {
    setSmoothing((current) => {
      if (current === "none") return "subtle";
      if (current === "subtle") return "balanced";
      if (current === "balanced") return "smooth";
      return "none";
    });
  };

  const showTexture = materialMode === "texture";
  const textureStatusLabel =
    textureState === "loading" ? "Loading…" :
    textureState === "loaded" ? "RGB texture" :
    textureState === "failed" ? "Texture failed" :
    "RGB texture";

  return (
    <div ref={wrapperRef} className={`terrain-viewer${isFullscreen ? " is-fullscreen" : ""}`}>
      <div className="terrain-toolbar" role="group" aria-label="3D terrain controls">
        <div className="terrain-toolbar__group">
          <button
            type="button"
            className={`viewer-control${showTexture ? " is-active" : ""}${textureState === "failed" ? " is-error" : ""}`}
            onClick={() => setMaterialMode("texture")}
            disabled={!textureUrl}
            aria-pressed={showTexture}
            aria-label="Show RGB satellite texture"
            title="Satellite photorealistic texture"
          >
            {textureState === "loading" ? <Loader size={14} className="spin-icon" /> : <ImageIcon size={14} />}
            <span>{textureStatusLabel}</span>
          </button>
          <button
            type="button"
            className={`viewer-control${!showTexture ? " is-active" : ""}`}
            onClick={() => setMaterialMode("height")}
            aria-pressed={!showTexture}
            aria-label="Show height elevation colormap"
            title="Elevation color gradient"
          >
            <Palette size={14} />
            <span>Height colors</span>
          </button>
          {textureState === "failed" ? (
            <button
              type="button"
              className="viewer-control viewer-control--retry"
              onClick={handleRetryTexture}
              aria-label="Retry texture loading"
            >
              <RefreshCw size={13} />
              <span>Retry</span>
            </button>
          ) : null}
          <button
            type="button"
            className={`viewer-control${smoothing !== "none" ? " is-active" : ""}`}
            onClick={cycleSmoothing}
            aria-label="Cycle terrain surface smoothing filter"
            title={`Surface Filter: ${smoothing.toUpperCase()} (Click to toggle smoothing level)`}
          >
            <Sparkles size={14} />
            <span>Smooth: {smoothing}</span>
          </button>
          <button
            type="button"
            className={`viewer-control${detrend ? " is-active" : ""}`}
            onClick={() => setDetrend((v) => !v)}
            aria-pressed={detrend}
            aria-label="Toggle local relief detrending filter"
            title="Remove perspective tilt / slope"
          >
            <Wand2 size={14} />
            <span>Detrend</span>
          </button>
          <button
            type="button"
            className={`viewer-control${withSkirt ? " is-active" : ""}`}
            onClick={() => setWithSkirt((v) => !v)}
            aria-pressed={withSkirt}
            aria-label="Toggle solid terrain 3D block base"
            title="Solid 3D block base"
          >
            <Layers size={14} />
            <span>3D Block</span>
          </button>
          <button
            type="button"
            className={`viewer-control${wireframe ? " is-active" : ""}`}
            onClick={() => setWireframe((current) => !current)}
            aria-pressed={wireframe}
            aria-label="Toggle terrain wireframe"
            title="Show triangular polygon mesh"
          >
            <Grid3X3 size={14} />
            <span>Wireframe</span>
          </button>
        </div>

        <label className="exaggeration-control" title="Adjust 3D height scale">
          <span>Relief</span>
          <input
            type="range"
            min="0.1"
            max="2.5"
            step="0.05"
            value={exaggeration}
            onChange={(event) => setExaggeration(Number(event.target.value))}
            aria-label="Terrain height exaggeration"
          />
          <output>{exaggeration.toFixed(2)}×</output>
        </label>

        <div className="terrain-toolbar__group terrain-toolbar__group--end">
          <button
            type="button"
            className={`viewer-control viewer-control--icon${autoRotate ? " is-active" : ""}`}
            onClick={() => setAutoRotate((v) => !v)}
            aria-label="Auto-rotate 3D model"
            title="Cinematic Orbit / Auto-rotate"
          >
            <Play size={14} />
          </button>
          <button
            type="button"
            className={`viewer-control viewer-control--icon${cameraPreset === "topdown" ? " is-active" : ""}`}
            onClick={handleTopDown}
            aria-label={cameraPreset === "topdown" ? "Switch to perspective view" : "Switch to top-down view"}
            title="Top-down orthographic view"
          >
            <ArrowDown size={14} />
          </button>
          <button
            type="button"
            className="viewer-control viewer-control--icon"
            onClick={() => {
              setResetKey((key) => key + 1);
              setCameraPreset("perspective");
              setAutoRotate(false);
            }}
            aria-label="Reset 3D camera"
            title="Reset view"
          >
            <RotateCcw size={14} />
          </button>
          <button
            type="button"
            className="viewer-control viewer-control--icon"
            onClick={toggleFullscreen}
            aria-label={isFullscreen ? "Exit fullscreen" : "View fullscreen"}
            title="Fullscreen"
          >
            {isFullscreen ? <Minimize size={14} /> : <Expand size={14} />}
          </button>
        </div>
      </div>

      <div className="terrain-canvas" role="img" aria-label="Interactive photorealistic predicted terrain reconstruction">
        <Canvas
          camera={{ position: PERSPECTIVE_CAMERA_POSITION, fov: 42, near: 0.1, far: 100 }}
          dpr={[1, 2]}
          frameloop="demand"
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
          shadows
        >
          <color attach="background" args={["#050e17"]} />
          <fog attach="fog" args={["#050e17", 9, 22]} />

          {/* Sunlight & Studio Illumination */}
          <ambientLight intensity={0.48} />
          <hemisphereLight args={["#d4efff", "#14221b", 0.95]} />
          <directionalLight
            position={[5, 8, 4]}
            intensity={2.1}
            castShadow
            shadow-mapSize={[1024, 1024]}
            shadow-bias={-0.0001}
          />
          {/* Subtle warm rim light for crisp architectural relief */}
          <directionalLight position={[-4, 3, -3]} intensity={0.55} color="#ffd494" />

          <TerrainMesh
            data={terrain}
            grid={depthGrid}
            inputWidth={inputWidth}
            inputHeight={inputHeight}
            geospatial={geospatial}
            textureUrl={textureUrl}
            showTexture={showTexture}
            wireframe={wireframe}
            exaggeration={exaggeration}
            onPointSelected={setSample}
            onTextureStateChange={setTextureState}
            textureLoadKey={textureLoadKey}
          />

          <gridHelper args={[14, 28, "#1a4652", "#0c232a"]} position={[0, -0.62, 0]} />
          <CameraControls resetKey={resetKey} preset={cameraPreset} autoRotate={autoRotate} />
        </Canvas>
      </div>

      <div className="terrain-scale" aria-hidden="true">
        <span>{formatNumber(terrain.minimum)}</span>
        <div />
        <span>{formatNumber(terrain.maximum)}</span>
        <em>{valueLabel}</em>
      </div>

      {sample ? (
        <div className="point-inspector" role="status">
          <div className="point-inspector__icon"><Focus size={15} /></div>
          <div>
            <span>Selected sample</span>
            <strong>x {sample.pixelX} · y {sample.pixelY} · z {formatNumber(sample.value, 5)}</strong>
            <small>{valueLabel}</small>
            {sample.mapX !== undefined && sample.mapY !== undefined ? (
              <small>
                {sample.coordinateKind === "geographic" ? "Lon / lat" : "Map X / Y"}: {formatNumber(sample.mapX, 6)}, {formatNumber(sample.mapY, 6)}
              </small>
            ) : null}
          </div>
          <button type="button" className="icon-button" onClick={() => setSample(null)} aria-label="Clear selected point">×</button>
        </div>
      ) : (
        <div className="terrain-hint">Drag to orbit · Scroll to zoom · Click the 3D surface to inspect elevation</div>
      )}

      <div className="mesh-resolution">
        Mesh {terrain.columns} × {terrain.rows} · 3D Diorama Block
      </div>
    </div>
  );
}
