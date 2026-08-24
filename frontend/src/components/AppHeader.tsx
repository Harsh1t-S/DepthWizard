import { CircleDotDashed, RadioTower } from "lucide-react";
import { API_BASE_URL } from "../lib/api";

export function AppHeader() {
  let apiHost = API_BASE_URL;
  try {
    apiHost = new URL(API_BASE_URL).host;
  } catch {
    // Keep the configured value when it is not a fully qualified URL.
  }

  return (
    <header className="app-header">
      <div className="brand-lockup" role="group" aria-label="DepthWizard home">
        <div className="brand-mark" aria-hidden="true">
          <CircleDotDashed size={21} strokeWidth={1.8} />
          <span className="brand-mark__scan" />
        </div>
        <div>
          <div className="brand-name">DepthWizard</div>
          <div className="brand-subtitle">Terrain Intelligence</div>
        </div>
      </div>

      <div className="mission-label" role="group" aria-label="Project attribution">
        <span>SIH26175</span>
        <span className="mission-label__divider" />
        <span>ISRO</span>
      </div>

      <div className="api-indicator" title={API_BASE_URL}>
        <span className="api-indicator__pulse" aria-hidden="true" />
        <RadioTower size={14} aria-hidden="true" />
        <span className="api-indicator__label">API</span>
        <span className="api-indicator__host">{apiHost}</span>
      </div>
    </header>
  );
}
