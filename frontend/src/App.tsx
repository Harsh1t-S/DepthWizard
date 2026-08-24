import { AlertTriangle, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AppHeader } from "./components/AppHeader";
import { EmptyWorkspace } from "./components/EmptyWorkspace";
import { ResultsWorkspace } from "./components/ResultsWorkspace";
import { UploadPanel } from "./components/UploadPanel";
import { analyzeRaster, loadDemo } from "./lib/api";
import { ApiError, type AnalysisResponse, type LoadingAction } from "./types/api";

function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") return "Request cancelled.";
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return "The analysis request failed unexpectedly.";
}

export default function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [dsmFile, setDsmFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loadingAction, setLoadingAction] = useState<LoadingAction | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!loadingAction) {
      setElapsedSeconds(0);
      return;
    }
    const startedAt = Date.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 500);
    return () => window.clearInterval(timer);
  }, [loadingAction]);

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  const runRequest = async (
    action: LoadingAction,
    request: (signal: AbortSignal) => Promise<AnalysisResponse>,
  ) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setError(null);
    setLoadingAction(action);

    try {
      const nextResult = await request(controller.signal);
      setResult(nextResult);
    } catch (requestError) {
      if (!(requestError instanceof DOMException && requestError.name === "AbortError")) {
        setError(errorMessage(requestError));
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setLoadingAction(null);
      }
    }
  };

  const handleAnalyze = () => {
    if (!imageFile) {
      setError("Choose an RGB image or GeoTIFF before starting analysis.");
      return;
    }
    void runRequest("analyze", (signal) => analyzeRaster(imageFile, dsmFile, signal));
  };

  const handleLoadDemo = () => {
    void runRequest("demo", loadDemo);
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoadingAction(null);
  };

  return (
    <div className="app-frame">
      <AppHeader />

      {error ? (
        <div className="error-banner" role="alert">
          <AlertTriangle size={17} aria-hidden="true" />
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">
            <X size={16} />
          </button>
        </div>
      ) : null}

      <main className="app-main">
        <UploadPanel
          imageFile={imageFile}
          dsmFile={dsmFile}
          loadingAction={loadingAction}
          elapsedSeconds={elapsedSeconds}
          onImageChange={(file) => {
            setImageFile(file);
            setError(null);
          }}
          onDsmChange={(file) => {
            setDsmFile(file);
            setError(null);
          }}
          onAnalyze={handleAnalyze}
          onLoadDemo={handleLoadDemo}
          onCancel={handleCancel}
          onValidationError={setError}
        />

        <div className="workspace-column">
          {result ? <ResultsWorkspace result={result} /> : <EmptyWorkspace />}
        </div>
      </main>

      <footer className="app-footer">
        <span>DepthWizard · SIH26175</span>
        <p>RGB-only output is relative. Metric elevation requires an aligned reference, DEM, or GCP calibration.</p>
        <span>Local · Open source · Offline-ready after model download</span>
      </footer>
    </div>
  );
}
