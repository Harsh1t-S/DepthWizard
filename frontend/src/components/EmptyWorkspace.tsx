import { Box, Crosshair, Layers3, ScanLine } from "lucide-react";

const workflowSteps = [
  { icon: ScanLine, label: "Ingest", copy: "RGB orbital raster" },
  { icon: Layers3, label: "Infer", copy: "Dense relative depth" },
  { icon: Box, label: "Explore", copy: "Interactive surface" },
];

export function EmptyWorkspace() {
  return (
    <section className="empty-workspace" aria-labelledby="workspace-heading">
      <div className="workspace-grid" aria-hidden="true" />
      <div className="contour-stage" aria-hidden="true">
        <span className="contour contour--one" />
        <span className="contour contour--two" />
        <span className="contour contour--three" />
        <span className="contour contour--four" />
        <span className="contour-stage__axis contour-stage__axis--x" />
        <span className="contour-stage__axis contour-stage__axis--y" />
        <div className="contour-stage__core">
          <Crosshair size={27} />
        </div>
        <span className="coordinate coordinate--north">N 28°36′</span>
        <span className="coordinate coordinate--east">E 77°12′</span>
      </div>

      <div className="empty-workspace__copy">
        <div className="ready-pill"><span /> Analysis workspace ready</div>
        <h1 id="workspace-heading">Reveal terrain structure from a single orbital image.</h1>
        <p>
          DepthWizard converts satellite imagery into a dense relative-depth field, inspectable maps, and an interactive 3D surface—directly in your browser.
        </p>
      </div>

      <div className="workflow-strip" aria-label="Analysis workflow">
        {workflowSteps.map(({ icon: Icon, label, copy }, index) => (
          <div className="workflow-step" key={label}>
            <div className="workflow-step__number">0{index + 1}</div>
            <Icon size={18} aria-hidden="true" />
            <div>
              <strong>{label}</strong>
              <span>{copy}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
