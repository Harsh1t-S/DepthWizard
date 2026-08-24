import {
  ArrowRight,
  Database,
  FileImage,
  ImagePlus,
  LoaderCircle,
  Play,
  Satellite,
  UploadCloud,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { formatBytes } from "../lib/format";
import type { LoadingAction } from "../types/api";

const ACCEPTED_RASTERS = ".jpg,.jpeg,.png,.tif,.tiff,image/jpeg,image/png,image/tiff";
const ACCEPTED_EXTENSION = /\.(?:jpe?g|png|tiff?)$/i;

interface UploadPanelProps {
  imageFile: File | null;
  dsmFile: File | null;
  loadingAction: LoadingAction | null;
  elapsedSeconds: number;
  onImageChange: (file: File | null) => void;
  onDsmChange: (file: File | null) => void;
  onAnalyze: () => void;
  onLoadDemo: () => void;
  onCancel: () => void;
  onValidationError: (message: string) => void;
}

function isAcceptedRaster(file: File): boolean {
  return ACCEPTED_EXTENSION.test(file.name);
}

function RasterFileRow({ file, onRemove, label }: { file: File; onRemove: () => void; label: string }) {
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
      <button type="button" className="icon-button" onClick={onRemove} aria-label={`Remove ${file.name}`}>
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
  loadingAction,
  elapsedSeconds,
  onImageChange,
  onDsmChange,
  onAnalyze,
  onLoadDemo,
  onCancel,
  onValidationError,
}: UploadPanelProps) {
  const imageInputRef = useRef<HTMLInputElement>(null);
  const dsmInputRef = useRef<HTMLInputElement>(null);
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

  const acceptFile = (file: File | undefined, kind: "image" | "dsm") => {
    if (!file) return;
    if (!isAcceptedRaster(file)) {
      onValidationError("Choose a JPG, PNG, TIF, or TIFF raster.");
      return;
    }
    if (kind === "image") onImageChange(file);
    else onDsmChange(file);
  };

  const busy = loadingAction !== null;

  return (
    <aside className="upload-panel" aria-label="Analysis input">
      <div className="section-kicker">
        <span>01</span>
        Scene input
      </div>
      <h2>Prepare an orbital scene</h2>
      <p className="panel-intro">Upload a single RGB satellite tile. Add an aligned DSM only when reference calibration is available.</p>

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

      <div className="optional-input">
        <div className="optional-input__heading">
          <div>
            <Database size={15} aria-hidden="true" />
            <span>Aligned DSM</span>
          </div>
          <span className="optional-chip">Optional</span>
        </div>
        <p>Enables scale alignment and reference metrics when supported by the scene.</p>
        <input
          ref={dsmInputRef}
          className="visually-hidden"
          type="file"
          accept={ACCEPTED_RASTERS}
          onChange={(event) => acceptFile(event.target.files?.[0], "dsm")}
          aria-label="Choose aligned DSM"
        />
        {dsmFile ? (
          <RasterFileRow file={dsmFile} label="reference DSM" onRemove={() => onDsmChange(null)} />
        ) : (
          <button
            type="button"
            className="secondary-button secondary-button--full"
            onClick={() => dsmInputRef.current?.click()}
            disabled={busy}
          >
            <ImagePlus size={16} />
            Add reference raster
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
          <strong>Relative by default.</strong> Monocular output encodes terrain structure, not absolute surveyed elevation. A reference DSM may calibrate scale.
        </p>
      </div>
    </aside>
  );
}
