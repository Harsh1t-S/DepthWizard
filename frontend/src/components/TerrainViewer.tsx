import { OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { ArrowDown, Expand, Focus, Grid3X3, Image as ImageIcon, Loader, Minimize, Palette, RefreshCw, RotateCcw } from "lucide-react";
import { type ElementRef, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { formatNumber } from "../lib/format";
import type { DepthGrid, GeospatialMetadata } from "../types/api";

const MAX_MESH_DIMENSION = 256;
const PERSPECTIVE_CAMERA_POSITION: [number, number, number] = [5.2, 4.4, 5.8];
const TOPDOWN_CAMERA_POSITION: [number, number, number] = [0, 8.5, 0];
const HEIGHT_SCALE = 0.6;

type TextureState = "idle" | "loading" | "loaded" | "failed";

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

function terrainColor(normalized: number): THREE.Color {
  const stops = [
    { at: 0, color: new THREE.Color("#10263d") },
    { at: 0.34, color: new THREE.Color("#176a80") },
    { at: 0.68, color: new THREE.Color("#4cc7a8") },
    { at: 1, color: new THREE.Color("#f0cf72") },
  ];
  const upperIndex = stops.findIndex((stop) => normalized <= stop.at);
  if (upperIndex <= 0) return stops[0].color.clone();
  const lower = stops[upperIndex - 1];
  const upper = stops[upperIndex];
  const amount = (normalized - lower.at) / (upper.at - lower.at || 1);
  return lower.color.clone().lerp(upper.color, amount);
}

function createTerrainGeometry(grid: DepthGrid): TerrainData {
  const dimensions = getGridDimensions(grid);
  const columns = Math.min(MAX_MESH_DIMENSION, dimensions.width);
  const rows = Math.min(MAX_MESH_DIMENSION, dimensions.height);
  const samples = new Float32Array(columns * rows);
  const validSamples = new Uint8Array(columns * rows);
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;

  for (let row = 0; row < rows; row += 1) {
    const sourceY = Math.round((row / Math.max(1, rows - 1)) * (dimensions.height - 1));
    for (let column = 0; column < columns; column += 1) {
      const sourceX = Math.round((column / Math.max(1, columns - 1)) * (dimensions.width - 1));
      const rawValue = getGridValue(grid, sourceX, sourceY);
      const value = Number.isFinite(rawValue) ? rawValue : 0;
      const index = row * columns + column;
      const valid = Number.isFinite(rawValue) && isGridValueValid(grid, sourceX, sourceY);
      samples[index] = value;
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

  const range = maximum - minimum || 1;
  // Skip re-normalization when backend already normalized values to [0,1]
  const alreadyNormalized = minimum >= -0.01 && maximum <= 1.01 && range <= 1.02;
  const sourceAspect = dimensions.width / Math.max(1, dimensions.height);
  const width = sourceAspect >= 1 ? 5.6 : 5.6 * sourceAspect;
  const depth = sourceAspect >= 1 ? 5.6 / sourceAspect : 5.6;
  const positions = new Float32Array(columns * rows * 3);
  const colors = new Float32Array(columns * rows * 3);
  const uvs = new Float32Array(columns * rows * 2);
  const indices: number[] = [];

  for (let row = 0; row < rows; row += 1) {
    const v = row / Math.max(1, rows - 1);
    for (let column = 0; column < columns; column += 1) {
      const u = column / Math.max(1, columns - 1);
      const index = row * columns + column;
      const normalized = alreadyNormalized
        ? Math.max(0, Math.min(1, samples[index]))
        : (samples[index] - minimum) / range;
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
  if (
    !geospatial
    || !geospatial.crs
    || geospatial.valid_for_dsm_export !== true
  ) return {};
  const transform = geospatial.transform;
  let mapX: number | undefined;
  let mapY: number | undefined;

  if (Array.isArray(transform) && transform.length >= 6 && transform.slice(0, 6).every(Number.isFinite)) {
    // Rasterio affine convention: x = a·col + b·row + c, y = d·col + e·row + f.
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

type CameraPreset = "perspective" | "topdown";

function CameraControls({ resetKey, preset }: { resetKey: number; preset: CameraPreset }) {
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
      dampingFactor={0.075}
      minDistance={2.7}
      maxDistance={13}
      maxPolarAngle={Math.PI / 2.02}
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
        nextTexture.anisotropy = 4;
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
        roughness={0.78}
        metalness={0.04}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

export default function TerrainViewer({ depthGrid, textureUrl, geospatial, inputWidth, inputHeight, valueLabel }: TerrainViewerProps) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const terrain = useMemo(() => createTerrainGeometry(depthGrid), [depthGrid]);
  const [materialMode, setMaterialMode] = useState<"texture" | "height">(textureUrl ? "texture" : "height");
  const [wireframe, setWireframe] = useState(false);
  const [exaggeration, setExaggeration] = useState(0.4);
  const [resetKey, setResetKey] = useState(0);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("perspective");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [sample, setSample] = useState<PointSample | null>(null);
  const [textureState, setTextureState] = useState<TextureState>("idle");
  const [textureLoadKey, setTextureLoadKey] = useState(0);

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
            aria-label="Show RGB texture"
          >
            {textureState === "loading" ? <Loader size={15} className="spin-icon" /> : <ImageIcon size={15} />}
            <span>{textureStatusLabel}</span>
          </button>
          <button
            type="button"
            className={`viewer-control${!showTexture ? " is-active" : ""}`}
            onClick={() => setMaterialMode("height")}
            aria-pressed={!showTexture}
            aria-label="Show height colors"
          >
            <Palette size={15} />
            <span>Height colors</span>
          </button>
          {textureState === "failed" ? (
            <button
              type="button"
              className="viewer-control viewer-control--retry"
              onClick={handleRetryTexture}
              aria-label="Retry texture loading"
            >
              <RefreshCw size={14} />
              <span>Retry</span>
            </button>
          ) : null}
          <button
            type="button"
            className={`viewer-control${wireframe ? " is-active" : ""}`}
            onClick={() => setWireframe((current) => !current)}
            aria-pressed={wireframe}
            aria-label="Toggle terrain wireframe"
          >
            <Grid3X3 size={15} />
            <span>Wireframe</span>
          </button>
        </div>
        <label className="exaggeration-control">
          <span>Height</span>
          <input
            type="range"
            min="0.1"
            max="3"
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
            className={`viewer-control viewer-control--icon${cameraPreset === "topdown" ? " is-active" : ""}`}
            onClick={handleTopDown}
            aria-label={cameraPreset === "topdown" ? "Switch to perspective view" : "Switch to top-down view"}
            title="Top-down view"
          >
            <ArrowDown size={15} />
          </button>
          <button type="button" className="viewer-control viewer-control--icon" onClick={() => { setResetKey((key) => key + 1); setCameraPreset("perspective"); }} aria-label="Reset 3D camera">
            <RotateCcw size={15} />
          </button>
          <button type="button" className="viewer-control viewer-control--icon" onClick={toggleFullscreen} aria-label={isFullscreen ? "Exit fullscreen" : "View fullscreen"}>
            {isFullscreen ? <Minimize size={15} /> : <Expand size={15} />}
          </button>
        </div>
      </div>

      <div className="terrain-canvas" role="img" aria-label="Interactive predicted terrain reconstruction">
        <Canvas
          camera={{ position: PERSPECTIVE_CAMERA_POSITION, fov: 43, near: 0.1, far: 100 }}
          dpr={[1, 1.7]}
          frameloop="demand"
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
          shadows
        >
          <color attach="background" args={["#061019"]} />
          <fog attach="fog" args={["#061019", 8, 17]} />
          <ambientLight intensity={0.62} />
          <hemisphereLight args={["#99e9ff", "#14251f", 1.1]} />
          <directionalLight position={[4, 7, 3]} intensity={2.2} castShadow />
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
          <gridHelper args={[12, 24, "#1c5160", "#102932"]} position={[0, -0.55, 0]} />
          <CameraControls resetKey={resetKey} preset={cameraPreset} />
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
        <div className="terrain-hint">Drag to orbit · Scroll to zoom · Click the surface to inspect</div>
      )}

      <div className="mesh-resolution">Mesh {terrain.columns} × {terrain.rows}</div>
    </div>
  );
}
