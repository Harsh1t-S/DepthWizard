export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
}

export function formatNumber(value: unknown, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return String(value ?? "—");
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute < 0.001 || absolute >= 100_000)) {
    return value.toExponential(2);
  }
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: digits }).format(value);
}

export function humanizeKey(key: string): string {
  const known: Record<string, string> = {
    abs_rel: "Abs Rel",
    sq_rel: "Sq Rel",
    rmse: "RMSE",
    mae: "MAE",
    r2: "R²",
    delta1: "δ < 1.25",
    delta2: "δ < 1.25²",
    delta3: "δ < 1.25³",
    ground_truth: "Ground truth",
    depth_grid: "Depth grid",
  };
  return known[key] || key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
