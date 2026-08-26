import { ApiError, type AnalysisResponse, type InferenceQualityMode } from "../types/api";

const configuredUrl = import.meta.env.VITE_API_URL?.trim();

// Empty means same-origin. The standalone build is served by the FastAPI
// process itself, and Vite proxies /api and /artifacts in development, so
// relative paths are correct in both. VITE_API_URL overrides this for a split
// deployment where the UI and API sit on different hosts.
export const API_BASE_URL = (configuredUrl || "").replace(/\/+$/, "");

function endpoint(path: string): string {
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

/** Absolute base used to resolve artifact URLs; falls back to the page origin. */
function absoluteBase(): string {
  if (API_BASE_URL) return `${API_BASE_URL}/`;
  return typeof window === "undefined" ? "http://localhost/" : `${window.location.origin}/`;
}

async function parseError(response: Response): Promise<ApiError> {
  let detail = "";

  try {
    const payload = (await response.json()) as {
      detail?: string | { msg?: string }[];
      message?: string;
      error?: string;
    };
    if (typeof payload.detail === "string") {
      detail = payload.detail;
    } else if (Array.isArray(payload.detail)) {
      detail = payload.detail.map((item) => item.msg).filter(Boolean).join("; ");
    } else {
      detail = payload.message || payload.error || "";
    }
  } catch {
    detail = (await response.text().catch(() => "")) || "";
  }

  const friendly =
    response.status === 413
      ? "The raster is larger than the server accepts. Try a smaller image."
      : response.status === 503
        ? "The inference service is warming up or the model is unavailable."
        : response.status >= 500
          ? "The analysis service encountered an error."
          : "The server could not process this request.";

  return new ApiError(detail || friendly, response.status, detail || undefined);
}

async function readAnalysisResponse(response: Response): Promise<AnalysisResponse> {
  if (!response.ok) {
    throw await parseError(response);
  }

  const payload = (await response.json()) as AnalysisResponse;
  if (!payload?.depth_grid || !payload?.input || !payload?.urls) {
    throw new ApiError("The server returned an incomplete analysis result.");
  }

  return payload;
}

export async function analyzeRaster(
  image: File,
  dsm: File | null,
  reference: File | null,
  qualityMode: InferenceQualityMode,
  acquisitionTime: string,
  signal?: AbortSignal,
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("image", image, image.name);
  if (dsm) {
    formData.append("ground_truth_dsm", dsm, dsm.name);
  }
  if (reference) {
    const field = /\.(?:csv|json)$/i.test(reference.name) ? "gcps" : "reference_dem";
    formData.append(field, reference, reference.name);
  }
  formData.append("quality_mode", qualityMode);
  // Enables shadow-based metric calibration on a georeferenced raster. The
  // backend ignores it for imagery with no CRS, and declines with a reason
  // rather than guessing when the solar geometry is unusable.
  //
  // Sent verbatim as UTC. Passing it through `new Date()` would reinterpret the
  // value in the viewer's timezone, so a scene captured at midday would arrive
  // as a time when the sun is below the horizon for any viewer far enough east
  // or west, and calibration would refuse.
  if (acquisitionTime.trim()) {
    // datetime-local yields "YYYY-MM-DDTHH:MM", sometimes with seconds.
    const withSeconds = /T\d{2}:\d{2}:\d{2}/.test(acquisitionTime)
      ? acquisitionTime
      : `${acquisitionTime}:00`;
    formData.append("acquisition_time", `${withSeconds}Z`);
  }

  try {
    const response = await fetch(endpoint("/api/analyze"), {
      method: "POST",
      body: formData,
      signal,
    });
    return await readAnalysisResponse(response);
  } catch (error) {
    if (error instanceof ApiError || (error instanceof DOMException && error.name === "AbortError")) {
      throw error;
    }
    throw new ApiError(`Cannot reach the DepthWizard API at ${API_BASE_URL || absoluteBase()}.`);
  }
}

export async function loadDemo(signal?: AbortSignal): Promise<AnalysisResponse> {
  try {
    const response = await fetch(endpoint("/api/demo"), { signal });
    return await readAnalysisResponse(response);
  } catch (error) {
    if (error instanceof ApiError || (error instanceof DOMException && error.name === "AbortError")) {
      throw error;
    }
    throw new ApiError(`Cannot reach the DepthWizard API at ${API_BASE_URL || absoluteBase()}.`);
  }
}

export function resolveApiUrl(value: string | null | undefined): string | null {
  if (!value) return null;

  try {
    return new URL(value, absoluteBase()).toString();
  } catch {
    return value;
  }
}
