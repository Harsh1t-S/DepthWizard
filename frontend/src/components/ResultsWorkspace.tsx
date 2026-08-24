import {
  AlertCircle,
  Box,
  Check,
  Clipboard,
  Clock3,
  Cpu,
  Download,
  FileImage,
  Gauge,
  Image as ImageIcon,
  Info,
  Layers3,
  Map,
  Maximize2,
  Satellite,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { resolveApiUrl } from "../lib/api";
import { formatDuration, formatNumber, humanizeKey } from "../lib/format";
import type { AnalysisResponse, DepthGrid } from "../types/api";

const TerrainViewer = lazy(() => import("./TerrainViewer"));

type ViewId = "original" | "depth" | "terrain" | "groundTruth" | "error";

interface ViewTab {
  id: ViewId;
  label: string;
  shortLabel: string;
  icon: typeof ImageIcon;
}

interface ResultsWorkspaceProps {
  result: AnalysisResponse;
}

function flattenValue(grid: DepthGrid, x: number, y: number): number {
  if (Array.isArray(grid.values[0])) {
    return Number((grid.values as number[][])[y]?.[x]);
  }
  return Number((grid.values as number[])[y * grid.width + x]);
}

function colorizeDepth(normalized: number): [number, number, number] {
  const stops: [number, number, number, number][] = [
    [0, 9, 20, 38],
    [0.24, 26, 63, 102],
    [0.48, 24, 139, 141],
    [0.72, 91, 201, 147],
    [1, 246, 211, 112],
  ];
  const upperIndex = stops.findIndex(([at]) => normalized <= at);
  if (upperIndex <= 0) return [stops[0][1], stops[0][2], stops[0][3]];
  const lower = stops[upperIndex - 1];
  const upper = stops[upperIndex];
  const amount = (normalized - lower[0]) / (upper[0] - lower[0] || 1);
  return [
    Math.round(lower[1] + (upper[1] - lower[1]) * amount),
    Math.round(lower[2] + (upper[2] - lower[2]) * amount),
    Math.round(lower[3] + (upper[3] - lower[3]) * amount),
  ];
}

function DepthGridCanvas({ grid }: { grid: DepthGrid }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const nested = Array.isArray(grid.values[0]);
    const sourceHeight = nested ? (grid.values as number[][]).length : grid.height;
    const sourceWidth = nested ? (grid.values as number[][])[0]?.length || grid.width : grid.width;
    const maxCanvasDimension = 900;
    const scale = Math.min(1, maxCanvasDimension / Math.max(sourceWidth, sourceHeight));
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));
    canvas.width = width;
    canvas.height = height;

    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    const values = grid.values.flat() as number[];
    for (const rawValue of values) {
      const value = Number(rawValue);
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const range = Number.isFinite(maximum - minimum) && maximum !== minimum ? maximum - minimum : 1;
    const context = canvas.getContext("2d");
    if (!context) return;
    const imageData = context.createImageData(width, height);

    for (let y = 0; y < height; y += 1) {
      const sourceY = Math.round((y / Math.max(1, height - 1)) * (sourceHeight - 1));
      for (let x = 0; x < width; x += 1) {
        const sourceX = Math.round((x / Math.max(1, width - 1)) * (sourceWidth - 1));
        const rawValue = flattenValue(grid, sourceX, sourceY);
        const normalized = Number.isFinite(rawValue) ? Math.max(0, Math.min(1, (rawValue - minimum) / range)) : 0;
        const [red, green, blue] = colorizeDepth(normalized);
        const index = (y * width + x) * 4;
        imageData.data[index] = red;
        imageData.data[index + 1] = green;
        imageData.data[index + 2] = blue;
        imageData.data[index + 3] = 255;
      }
    }
    context.putImageData(imageData, 0, 0);
  }, [grid]);

  return (
    <div className="grid-canvas-wrap">
      <canvas ref={canvasRef} aria-label="Predicted relative depth raster" />
      <div className="raster-color-scale" aria-label="Relative depth color scale">
        <span>Low</span>
        <i />
        <span>High</span>
      </div>
    </div>
  );
}

function RasterView({ src, alt, fallbackGrid }: { src: string | null; alt: string; fallbackGrid?: DepthGrid }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);

  if ((!src || failed) && fallbackGrid) return <DepthGridCanvas grid={fallbackGrid} />;
  if (!src || failed) {
    return (
      <div className="raster-unavailable" role="status">
        <FileImage size={31} />
        <strong>Preview unavailable</strong>
        <span>The artifact can still be available from the downloads panel.</span>
      </div>
    );
  }

  return (
    <div className="raster-stage">
      <div className="raster-stage__grid" aria-hidden="true" />
      <img src={src} alt={alt} onError={() => setFailed(true)} decoding="async" />
    </div>
  );
}

function ResultBadge({ result }: { result: AnalysisResponse }) {
  if (result.demo) {
    return <span className="result-badge result-badge--demo"><Sparkles size={13} /> Precomputed synthetic demo</span>;
  }
  if (result.precomputed) {
    return <span className="result-badge result-badge--benchmark"><ShieldCheck size={13} /> Precomputed result</span>;
  }
  return <span className="result-badge result-badge--live"><span /> Live inference</span>;
}

function DataTile({ icon: Icon, label, value, detail }: { icon: typeof Cpu; label: string; value: string; detail?: string }) {
  return (
    <div className="data-tile">
      <div className="data-tile__icon" aria-hidden="true"><Icon size={16} /></div>
      <div>
        <span>{label}</span>
        <strong title={value}>{value}</strong>
        {detail ? <small>{detail}</small> : null}
      </div>
    </div>
  );
}

function formatMetric(key: string, value: number | string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (key.startsWith("delta") && Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`;
  return formatNumber(value, 4);
}

function MetricsPanel({ result }: { result: AnalysisResponse }) {
  const metricEntries = result.metrics
    ? Object.entries(result.metrics).filter((entry): entry is [string, string | number] => typeof entry[1] === "number" || typeof entry[1] === "string")
    : [];
  const calibrationEntries = result.calibration
    ? Object.entries(result.calibration).filter(([, value]) => value !== null && value !== undefined && typeof value !== "object")
    : [];

  return (
    <section className="details-card details-card--metrics">
      <div className="details-card__heading">
        <div>
          <span className="details-card__eyebrow">Reference evaluation</span>
          <h3>Benchmark calibration / feasibility evaluation</h3>
        </div>
        {result.calibration ? <span className="calibrated-chip"><Check size={12} /> Reference-aligned</span> : null}
      </div>

      {metricEntries.length > 0 ? (
        <div className="metric-grid">
          {metricEntries.map(([key, value]) => (
            <div className="metric-item" key={key}>
              <span>{humanizeKey(key)}</span>
              <strong>{formatMetric(key, value)}</strong>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-reference">
          <Gauge size={21} />
          <div>
            <strong>No reference metrics for this scene</strong>
            <p>Prediction remains a relative depth field until evaluated or aligned against a compatible DSM.</p>
          </div>
        </div>
      )}

      {calibrationEntries.length > 0 ? (
        <div className="calibration-strip" role="group" aria-label="Calibration coefficients" tabIndex={0}>
          <span>Calibration</span>
          {calibrationEntries.slice(0, 5).map(([key, value]) => (
            <div key={key}><em>{humanizeKey(key)}</em><strong>{formatNumber(value)}</strong></div>
          ))}
        </div>
      ) : null}

      <p className="evaluation-disclaimer">
        {result.demo
          ? "Metrics describe only this bundled synthetic fixture; they are not a real-world benchmark result."
          : "Metrics describe this supplied benchmark/reference pair. They do not certify general accuracy or convert uncalibrated output into absolute elevation."}
      </p>
    </section>
  );
}

function extractBoundsText(result: AnalysisResponse): string | null {
  const bounds = result.geospatial?.bounds;
  if (!bounds) return null;
  if (Array.isArray(bounds)) return bounds.slice(0, 4).map((value) => formatNumber(value, 4)).join(" · ");
  const values = [bounds.left ?? bounds.min_x, bounds.bottom ?? bounds.min_y, bounds.right ?? bounds.max_x, bounds.top ?? bounds.max_y];
  return values.every((value) => typeof value === "number") ? values.map((value) => formatNumber(value, 4)).join(" · ") : null;
}

function MetadataPanel({ result }: { result: AnalysisResponse }) {
  const boundsText = extractBoundsText(result);
  const crs = result.geospatial?.crs ?? (result.geospatial?.epsg ? `EPSG:${result.geospatial.epsg}` : null);
  const isGeoreferenced = Boolean(crs && result.geospatial?.valid_for_dsm_export === true);

  return (
    <section className="details-card details-card--metadata">
      <div className="details-card__heading">
        <div>
          <span className="details-card__eyebrow">Scene metadata</span>
          <h3>Spatial context</h3>
        </div>
        <Map size={18} aria-hidden="true" />
      </div>
      <dl className="metadata-list">
        <div><dt>Dimensions</dt><dd>{result.input.width.toLocaleString()} × {result.input.height.toLocaleString()} px</dd></div>
        <div><dt>CRS</dt><dd>{crs ? String(crs) : "Not embedded"}</dd></div>
        {isGeoreferenced && boundsText ? <div><dt>Bounds</dt><dd className="metadata-list__small">{boundsText}</dd></div> : null}
        <div><dt>Output basis</dt><dd>{result.calibration ? "Benchmark-calibrated height" : "Relative depth"}</dd></div>
      </dl>
      {!isGeoreferenced ? (
        <div className="metadata-note"><Info size={14} /><span>No georeferencing was preserved. Point inspection uses image pixels.</span></div>
      ) : null}
    </section>
  );
}

function ArtifactPanel({ result }: { result: AnalysisResponse }) {
  const artifacts = Object.entries(result.artifacts ?? {}).filter(([, value]) => Boolean(value));
  return (
    <section className="details-card details-card--artifacts">
      <div className="details-card__heading">
        <div>
          <span className="details-card__eyebrow">Export</span>
          <h3>Artifacts</h3>
        </div>
        <Download size={18} aria-hidden="true" />
      </div>
      {artifacts.length > 0 ? (
        <div className="artifact-list">
          {artifacts.map(([name, value]) => {
            const href = resolveApiUrl(value);
            return href ? (
              <a href={href} target="_blank" rel="noreferrer" download key={name}>
                <span className="artifact-list__icon"><Download size={14} /></span>
                <span><strong>{humanizeKey(name)}</strong><small>Download file</small></span>
              </a>
            ) : null;
          })}
        </div>
      ) : (
        <p className="empty-artifacts">No downloadable artifacts were returned for this run.</p>
      )}
    </section>
  );
}

export function ResultsWorkspace({ result }: ResultsWorkspaceProps) {
  const resolvedUrls = useMemo(
    () => ({
      original: resolveApiUrl(result.urls.original),
      depth: resolveApiUrl(result.urls.depth),
      groundTruth: resolveApiUrl(result.urls.ground_truth),
      error: resolveApiUrl(result.urls.error),
    }),
    [result.urls.depth, result.urls.error, result.urls.ground_truth, result.urls.original],
  );
  const tabs = useMemo<ViewTab[]>(() => {
    const nextTabs: ViewTab[] = [];
    if (resolvedUrls.original) nextTabs.push({ id: "original", label: "Original", shortLabel: "Original", icon: ImageIcon });
    nextTabs.push({ id: "depth", label: "Predicted Depth", shortLabel: "Depth", icon: Layers3 });
    nextTabs.push({ id: "terrain", label: "3D Reconstruction", shortLabel: "3D", icon: Box });
    if (resolvedUrls.groundTruth) nextTabs.push({ id: "groundTruth", label: "Ground Truth", shortLabel: "Reference", icon: Satellite });
    if (resolvedUrls.error) nextTabs.push({ id: "error", label: "Error Map", shortLabel: "Error", icon: AlertCircle });
    return nextTabs;
  }, [resolvedUrls.error, resolvedUrls.groundTruth, resolvedUrls.original]);
  const [activeView, setActiveView] = useState<ViewId>("depth");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setActiveView("depth");
    setCopied(false);
  }, [result.job_id]);

  const copyJobId = async () => {
    try {
      await navigator.clipboard.writeText(result.job_id);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };

  const imageAlt: Record<Exclude<ViewId, "terrain">, string> = {
    original: `Original satellite image ${result.input.filename}`,
    depth: "Predicted relative depth map",
    groundTruth: "Ground-truth DSM visualization",
    error: "Prediction error map against the reference DSM",
  };

  return (
    <section className="results-workspace" aria-labelledby="result-heading">
      <div className="results-header">
        <div>
          <div className="results-header__topline">
            <ResultBadge result={result} />
            <span className="results-header__mode">{result.mode}</span>
          </div>
          <h1 id="result-heading">Terrain analysis complete</h1>
          <p title={result.input.filename}>{result.input.filename}</p>
        </div>
        <button type="button" className="job-id" onClick={copyJobId} aria-label="Copy job ID">
          <span>Job</span>
          <code>{result.job_id.slice(0, 12)}</code>
          {copied ? <Check size={14} /> : <Clipboard size={14} />}
        </button>
      </div>

      <div className="run-summary" role="group" aria-label="Run summary">
        <DataTile icon={ImageIcon} label="Input raster" value={`${result.input.width.toLocaleString()} × ${result.input.height.toLocaleString()}`} detail="pixels" />
        <DataTile icon={Clock3} label="Processing" value={formatDuration(result.processing_time_seconds)} detail="end to end" />
        <DataTile icon={Cpu} label="Compute" value={result.device || "Unknown"} detail={result.model || "model not reported"} />
        <DataTile icon={Gauge} label="Depth basis" value={result.calibration ? "Calibrated" : "Relative"} detail={result.calibration ? "reference-aligned" : "scale-ambiguous"} />
      </div>

      <section className="viewer-card" aria-label="Analysis visualizations">
        <div className="view-tabs" role="tablist" aria-label="Result views">
          {tabs.map(({ id, label, shortLabel, icon: Icon }) => (
            <button
              type="button"
              role="tab"
              id={`tab-${id}`}
              aria-selected={activeView === id}
              aria-controls="result-view-panel"
              className={activeView === id ? "is-active" : ""}
              onClick={() => setActiveView(id)}
              key={id}
            >
              <Icon size={15} aria-hidden="true" />
              <span className="tab-label-long">{label}</span>
              <span className="tab-label-short">{shortLabel}</span>
            </button>
          ))}
          <div className="view-tabs__meta">
            <Maximize2 size={13} />
            <span>{result.depth_grid.width} × {result.depth_grid.height} grid</span>
          </div>
        </div>

        <div className="view-panel" role="tabpanel" id="result-view-panel" aria-labelledby={`tab-${activeView}`}>
          {activeView === "terrain" ? (
            <Suspense fallback={<div className="viewer-loading"><span /><p>Building terrain mesh…</p></div>}>
              <TerrainViewer
                depthGrid={result.depth_grid}
                textureUrl={resolvedUrls.original}
                geospatial={result.geospatial}
                inputWidth={result.input.width}
                inputHeight={result.input.height}
                valueLabel={result.calibration ? "reference-aligned height" : "relative depth"}
              />
            </Suspense>
          ) : activeView === "depth" ? (
            <RasterView src={resolvedUrls.depth} alt={imageAlt.depth} fallbackGrid={result.depth_grid} />
          ) : activeView === "original" ? (
            <RasterView src={resolvedUrls.original} alt={imageAlt.original} />
          ) : activeView === "groundTruth" ? (
            <RasterView src={resolvedUrls.groundTruth} alt={imageAlt.groundTruth} />
          ) : (
            <RasterView src={resolvedUrls.error} alt={imageAlt.error} />
          )}
        </div>

        <div className="viewer-caption">
          <span className="viewer-caption__status"><span /> {activeView === "terrain" ? "Interactive surface" : "Rendered artifact"}</span>
          <p>
            {activeView === "depth"
              ? "Brighter values indicate larger model-relative depth values; they are not metres without reference calibration."
              : activeView === "terrain"
                ? result.calibration
                  ? "Surface shape is normalized for display; inspected values are benchmark reference-aligned heights."
                  : "Surface height visualizes normalized relative model output. Use exaggeration for inspection only."
                : activeView === "error"
                  ? "Error is computed only against the supplied reference scene."
                  : activeView === "groundTruth"
                    ? "Reference data supplied for benchmark evaluation and calibration."
                    : "Source imagery used for this analysis job."}
          </p>
        </div>
      </section>

      {result.notices?.length ? (
        <div className="notice-list" role="list" aria-label="Analysis notices">
          {result.notices.map((notice, index) => (
            <div role="listitem" key={`${index}-${notice}`}><Info size={15} /><span>{notice}</span></div>
          ))}
        </div>
      ) : null}

      <div className="details-grid">
        <MetricsPanel result={result} />
        <MetadataPanel result={result} />
        <ArtifactPanel result={result} />
      </div>
    </section>
  );
}
