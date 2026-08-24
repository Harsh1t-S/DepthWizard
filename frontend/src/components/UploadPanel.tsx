import {
  ArrowRight,
  Database,
  FileImage,
  ImagePlus,
  LoaderCircle,
  MapPin,
  Play,
  Ruler,
  Satellite,
  UploadCloud,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { formatBytes } from "../lib/format";
import type { LoadingAction } from "../types/api";

const ACCEPTED_RASTERS = ".jpg,.jpeg,.png,.tif,.tiff,image/jpeg,image/png,image/tiff";
const ACCEPTED_EXTENSION = /\.(?:jpe?g|png|tiff?)$/i;
const ACCEPTED_CALIBRATION = ".tif,.tiff,.npy,.csv,.json,image/tiff,text/csv,application/json,application/octet-stream";
const ACCEPTED_CALIBRATION_EXTENSION = /\.(?:tiff?|npy|csv|json)$/i;

interface UploadPanelProps {
  imageFile: File | null;
  dsmFile: File | null;
  referenceFile: File | null;
  loadingAction: LoadingAction | null;
  elapsedSeconds: number;
  onImageChange: (file: File | null) => void;
  onDsmChange: (file: File | null) => void;
  onReferenceChange: (file: File | null) => void;
  onAnalyze: () => void;
  onLoadDemo: () => void;
  onCancel: () => void;
  onValidationError: (message: string) => void;
}

function isAcceptedRaster(file: File): boolean {
  return ACCEPTED_EXTENSION.test(file.name);
}

function RasterFileRow({ file, onRemove, label, disabled = false }: { file: File; onRemove: () => void; label: string; disabled?: boolean }) {
  return (
    <div className="raster-file">
      <div className="raster-file__icon" aria-hidden="true">
        <FileImage size={17} />
      </div>
      <div className="raster-file__copy">
        <span className="raster-file__name" title={file.name}>
          {file.name}
        </span>
        <span className="raster-file__meta">
          {label} · {formatBytes(file.size)}
        </span>
      </div>
      <button type="button" className="icon-button" onClick={onRemove} aria-label={`Remove ${file.name}`} disabled={disabled}>
        <X size={16} />
      </button>
    </div>
  );
}

function LoadingState({ action, elapsed, onCancel }: { action: LoadingAction; elapsed: number; onCancel: () => void }) {
  const isDemo = action === "demo";
  const headline = isDemo
    ? "Retrieving demo scene"
    : elapsed < 2
      ? "Uploading raster"
      : elapsed < 8
        ? "Estimating terrain depth"
        : "Initializing inference model";

  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-state__topline">
        <div className="loading-state__spinner" aria-hidden="true">
          <LoaderCircle size={20} />
        </div>
        <div>
          <strong>{headline}</strong>
          <span>{Math.max(1, elapsed)} s elapsed</span>
        </div>
      </div>
      <div className="loading-track" aria-hidden="true">
        <span />
      </div>
      <p>
        {elapsed >= 8 && !isDemo
          ? "First run can take longer while model weights are downloaded and cached."
          : isDemo
            ? "Loading the clearly labeled synthetic demo—no local upload required."
            : "Keeping source resolution and metadata attached to this job."}
      </p>
      <button type="button" className="text-button" onClick={onCancel}>
        Cancel request
      </button>
    </div>
  );
}

export function UploadPanel({
  imageFile,
  dsmFile,
  referenceFile,
  loadingAction,
  elapsedSeconds,
  onImageChange,
  onDsmChange,
  onReferenceChange,
  onAnalyze,
  onLoadDemo,
  onCancel,
  onValidationError,
}: UploadPanelProps) {
  const imageInputRef = useRef<HTMLInputElement>(null);
  const dsmInputRef = useRef<HTMLInputElement>(null);
  const referenceInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!imageFile || !/\.(?:jpe?g|png)$/i.test(imageFile.name)) {
      setPreviewUrl(null);
      return;
    }

    const nextUrl = URL.createObjectURL(imageFile);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [imageFile]);

  const acceptFile = (file: File | undefined, kind: "image" | "dsm" | "reference") => {
    if (!file) return;
    if (kind === "reference" && !ACCEPTED_CALIBRATION_EXTENSION.test(file.name)) {
      onValidationError("Choose a TIF/TIFF/NPY reference DEM or a CSV/JSON GCP file.");
      return;
    }
    if (kind !== "reference" && !isAcceptedRaster(file)) {
      onValidationError("Choose a JPG, PNG, TIF, or TIFF raster.");
      return;
    }
    if (kind === "image") onImageChange(file);
    else if (kind === "dsm") onDsmChange(file);
    else onReferenceChange(file);
  };

  const busy = loadingAction !== null;

  return (
    <aside className="upload-panel" aria-label="Analysis input">
      <div className="section-kicker">
        <span>01</span>
        Scene input
      </div>
      <h2>Prepare an orbital scene</h2>
      <p className="panel-intro">Upload one RGB satellite tile, then optionally choose a deployment calibration source or benchmark ground truth.</p>

      <input
        ref={imageInputRef}
        className="visually-hidden"
        type="file"
        accept={ACCEPTED_RASTERS}
        onChange={(event) => acceptFile(event.target.files?.[0], "image")}
        aria-label="Choose satellite image"
      />

      {imageFile ? (
        <div className="selected-raster-preview">
          {previewUrl ? (
            <img src={previewUrl} alt="Selected satellite raster preview" />
          ) : (
            <div className="tiff-preview" aria-hidden="true">
              <Satellite size={29} />
              <span>GeoTIFF raster</span>
            </div>
          )}
          <div className="selected-raster-preview__shade" />
          <button
            type="button"
            className="selected-raster-preview__replace"
            onClick={() => imageInputRef.current?.click()}
            disabled={busy}
          >
            Replace image
          </button>
          <div className="selected-raster-preview__file">
            <strong title={imageFile.name}>{imageFile.name}</strong>
            <span>{formatBytes(imageFile.size)} · source image</span>
          </div>
        </div>
      ) : (
        <button
          type="button"
          className={`drop-zone${isDragging ? " is-dragging" : ""}`}
          onClick={() => imageInputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            acceptFile(event.dataTransfer.files?.[0], "image");
          }}
          disabled={busy}
        >
          <span className="drop-zone__icon" aria-hidden="true">
            <UploadCloud size={23} />
          </span>
          <strong>Drop satellite imagery</strong>
          <span>or click to browse</span>
          <small>JPG · PNG · TIF · TIFF</small>
        </button>
      )}

      <div className="optional-input optional-input--calibration">
        <div className="optional-input__heading">
          <div>
            <MapPin size={15} aria-hidden="true" />
            <span>Reference DEM / SRTM / GCP</span>
          </div>
          <span className="optional-chip optional-chip--calibration">Calibration</span>
        </div>
        <p>Deployment input for metric scale: a DEM raster/array or sparse control points.</p>
        <input
          ref={referenceInputRef}
          className="visually-hidden"
          type="file"
          accept={ACCEPTED_CALIBRATION}
          onChange={(event) => acceptFile(event.target.files?.[0], "reference")}
          aria-label="Choose reference DEM, SRTM, or GCP file"
        />
        {referenceFile ? (
          <RasterFileRow
            file={referenceFile}
            label={/\.(?:csv|json)$/i.test(referenceFile.name) ? "sparse GCP calibration" : "reference DEM calibration"}
            disabled={busy}
            onRemove={() => {
              if (referenceInputRef.current) referenceInputRef.current.value = "";
              onReferenceChange(null);
            }}
          />
        ) : (
          <button
            type="button"
            className="secondary-button secondary-button--full"
            onClick={() => referenceInputRef.current?.click()}
            disabled={busy}
          >
            <Ruler size={16} />
            Add DEM or control points
          </button>
        )}
        <small className="input-formats">DEM: TIF · TIFF · NPY&nbsp;&nbsp; GCP: CSV · JSON</small>
      </div>

      <div className="optional-input optional-input--benchmark">
        <div className="optional-input__heading">
          <div>
            <Database size={15} aria-hidden="true" />
            <span>Aligned DSM ground truth</span>
          </div>
          <span className="optional-chip optional-chip--benchmark">Benchmark only</span>
        </div>
        <p>Full-coverage target used to fit and score this scene. It is not a deployment calibration input.</p>
        <input
          ref={dsmInputRef}
          className="visually-hidden"
          type="file"
          accept={ACCEPTED_RASTERS}
          onChange={(event) => acceptFile(event.target.files?.[0], "dsm")}
          aria-label="Choose aligned DSM"
        />
        {dsmFile ? (
          <RasterFileRow
            file={dsmFile}
            label="full ground-truth DSM"
            disabled={busy}
            onRemove={() => {
              if (dsmInputRef.current) dsmInputRef.current.value = "";
              onDsmChange(null);
            }}
          />
        ) : (
          <button
            type="button"
            className="secondary-button secondary-button--full"
            onClick={() => dsmInputRef.current?.click()}
            disabled={busy}
          >
            <ImagePlus size={16} />
            Add benchmark ground truth
          </button>
        )}
      </div>

      {loadingAction ? (
        <LoadingState action={loadingAction} elapsed={elapsedSeconds} onCancel={onCancel} />
      ) : (
        <>
          <button type="button" className="primary-button" onClick={onAnalyze} disabled={!imageFile}>
            <span>Analyze terrain</span>
            <ArrowRight size={17} />
          </button>
          <div className="button-separator">
            <span />
            <em>or</em>
            <span />
          </div>
          <button type="button" className="demo-button" onClick={onLoadDemo}>
            <Play size={15} fill="currentColor" />
            Load demo scene
          </button>
        </>
      )}

      <div className="honesty-note">
        <span className="honesty-note__mark" aria-hidden="true">R</span>
        <p>
          <strong>RGB-only stays relative.</strong> DEM/GCP input produces a reference-calibrated metric estimate. A full aligned DSM is reserved for benchmark fitting.
        </p>
      </div>
    </aside>
  );
}
