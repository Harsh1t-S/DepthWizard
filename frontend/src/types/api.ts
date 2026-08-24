export type DepthValues = number[] | number[][];

export interface DepthGrid {
  width: number;
  height: number;
  values: DepthValues;
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

export interface ResultUrls {
  original?: string | null;
  depth?: string | null;
  ground_truth?: string | null;
  error?: string | null;
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
