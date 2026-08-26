export type DepthValues = number[] | number[][];

export interface BuildingFootprint {
  /** Clockwise normalized image coordinates, [u, v]. */
  points: [number, number][];
  roof_height: number;
  base_height: number;
  area_pixels?: number;
  confidence?: number;
}

/** Full-resolution height field, carried as a 16-bit PNG. */
export interface EncodedHeightGrid {
  url: string;
  format: "png16";
  minimum: number;
  maximum: number;
  width: number;
  height: number;
  mask_url?: string | null;
}

export interface DepthGrid {
  width: number;
  height: number;
  values: DepthValues;
  valid_mask?: boolean[] | boolean[][];
  minimum?: number;
  maximum?: number;
  /**
   * Present when the backend emitted a high-resolution height image. The
   * `values` array above stays as a coarse fallback, so a client that ignores
   * this still renders, just with softer edges.
   */
  encoded?: EncodedHeightGrid | null;
  /** Display-only roof polygons used to render true vertical building walls. */
  building_footprints?: BuildingFootprint[];
}

export interface InputMetadata {
  width: number;
  height: number;
  filename: string;
}

export interface BoundsObject {
  left?: number;
  bottom?: number;
  right?: number;
  top?: number;
  min_x?: number;
  min_y?: number;
  max_x?: number;
  max_y?: number;
}

export interface GeospatialMetadata {
  crs?: string | null;
  epsg?: string | number | null;
  transform?: number[] | null;
  bounds?: BoundsObject | number[] | null;
  pixel_size?: number | number[] | null;
  resolution?: number | number[] | null;
  units?: string | null;
  [key: string]: unknown;
}

export interface DepthMetrics {
  mae?: number;
  rmse?: number;
  abs_rel?: number;
  sq_rel?: number;
  delta1?: number;
  delta2?: number;
  delta3?: number;
  r2?: number;
  [key: string]: number | string | null | undefined;
}

export interface CalibrationMetadata {
  method?: string;
  scale?: number;
  shift?: number;
  units?: string;
  reference?: string;
  note?: string;
  [key: string]: unknown;
}

export interface ReferenceSummary {
  type?: string;
  source?: string;
  filename?: string;
  point_count?: number;
  calibration_points?: number;
  holdout_points?: number;
  coverage?: number;
  units?: string;
  [key: string]: string | number | boolean | null | undefined;
}

export type InferenceQualityMode = "fast" | "quality";

export interface InferenceSummary {
  quality_mode: InferenceQualityMode | string;
  passes: number;
  tiled: boolean;
  tile_count: number;
  bounded_width?: number;
  bounded_height?: number;
}

export interface ResultUrls {
  original?: string | null;
  depth?: string | null;
  reference_dem?: string | null;
  ground_truth?: string | null;
  error?: string | null;
  /** Full-resolution height field for the mesh, as a 16-bit PNG. */
  height16?: string | null;
  /** Tangent-space normals derived from the prediction, for lighting detail. */
  normal?: string | null;
}

export interface AnalysisResponse {
  job_id: string;
  demo?: boolean;
  precomputed?: boolean;
  model: string;
  device: string;
  mode: string;
  input: InputMetadata;
  processing_time_seconds: number;
  geospatial: GeospatialMetadata | null;
  metrics: DepthMetrics | null;
  calibration: CalibrationMetadata | null;
  reference?: ReferenceSummary | null;
  reference_summary?: ReferenceSummary | null;
  inference?: InferenceSummary | null;
  depth_grid: DepthGrid;
  urls: ResultUrls;
  artifacts: Record<string, string>;
  notices: string[];
}

export type LoadingAction = "analyze" | "demo";

export class ApiError extends Error {
  readonly status?: number;
  readonly detail?: string;

  constructor(message: string, status?: number, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}
