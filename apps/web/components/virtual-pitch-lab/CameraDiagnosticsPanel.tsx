import type { CameraBridgeDiagnostics, LabCameraBridgeInput } from "./types";
import type { ActiveCameraDiagnostics } from "@/components/virtual-pitch";


function metric(value?: number | null, suffix = ""): string {
  return value == null || !Number.isFinite(value) ? "Unavailable" : `${value.toFixed(3)}${suffix}`;
}


function pixel(point?: { x: number; y: number } | null): string {
  return point ? `${point.x.toFixed(2)}, ${point.y.toFixed(2)}` : "Unavailable";
}


export function CameraDiagnosticsPanel({
  camera,
  diagnostics,
  activeCamera,
  requestedCamera,
  activeCanvasCount
}: {
  camera: LabCameraBridgeInput | null;
  diagnostics?: CameraBridgeDiagnostics | null;
  activeCamera?: ActiveCameraDiagnostics | null;
  requestedCamera?: string;
  activeCanvasCount?: number;
}) {
  if (!camera) return null;
  const warnings = [...new Set([...(diagnostics?.warnings ?? []), ...(camera.warnings ?? [])])];
  const acceptedLabel = camera.accepted
    ? "Accepted Ground-Plane Calibration"
    : camera.source.toLowerCase().includes("synthetic")
      ? "Synthetic camera validation"
      : "Unaccepted Camera Candidate - Visual Validation Only";
  return (
    <div className="space-y-4">
      <div className="border-l-2 border-[#ffe56b] pl-3">
        <p className="text-xs font-black text-white">{acceptedLabel}</p>
        <p className="mt-1 break-words text-[11px] text-white/45">{camera.source} / {camera.candidate_id ?? "no candidate ID"}</p>
      </div>
      <dl className="grid grid-cols-2 gap-x-3 gap-y-3 text-xs">
        <div><dt className="text-white/35">Requested camera</dt><dd className="mt-1 font-semibold text-white/80">{requestedCamera ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Active camera</dt><dd className="mt-1 font-semibold text-white/80">{activeCamera?.activeCameraMode ?? "Preparing"}</dd></div>
        <div><dt className="text-white/35">Camera UUID</dt><dd className="mt-1 break-all font-mono text-[10px] text-white/80">{activeCamera?.cameraUuid ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Instances / canvases</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{activeCamera ? `${activeCamera.cameraInstanceCount} / ${activeCanvasCount ?? "?"}` : "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Camera ready</dt><dd className="mt-1 font-semibold text-white/80">{activeCamera?.cameraReady ? "Yes" : "No"}</dd></div>
        <div><dt className="text-white/35">Orbit controls</dt><dd className="mt-1 font-semibold text-white/80">{activeCamera?.orbitControlsMounted ? "Mounted" : "Unmounted"}</dd></div>
        <div><dt className="text-white/35">Custom projection</dt><dd className="mt-1 font-semibold text-white/80">{activeCamera?.customProjectionActive ? "Active" : "Inactive"}</dd></div>
        <div><dt className="text-white/35">Active-camera RMSE</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{metric(activeCamera?.activeCameraRmse, " px")}</dd></div>
        <div><dt className="text-white/35">Pose checksum</dt><dd className="mt-1 font-mono text-[10px] text-white/80">{activeCamera?.matrixWorldChecksum ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Projection checksum</dt><dd className="mt-1 font-mono text-[10px] text-white/80">{activeCamera?.projectionChecksum ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Displayed media</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{activeCamera ? `${Math.round(activeCamera.displayedMediaWidth)} x ${Math.round(activeCamera.displayedMediaHeight)}` : "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Classification</dt><dd className="mt-1 font-semibold text-white/80">{camera.classification}</dd></div>
        <div><dt className="text-white/35">Native frame</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{camera.image_width} x {camera.image_height}</dd></div>
        <div><dt className="text-white/35">Bridge RMSE</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{metric(diagnostics?.rmse_px, " px")}</dd></div>
        <div><dt className="text-white/35">Maximum error</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{metric(diagnostics?.maximum_error_px, " px")}</dd></div>
        <div><dt className="text-white/35">Landmarks</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{diagnostics ? `${diagnostics.valid_point_count}/${diagnostics.point_count}` : "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Invalid points</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{diagnostics?.invalid_point_count ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Distortion</dt><dd className="mt-1 break-words font-semibold text-white/80">{diagnostics?.distortion_mode ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Bridge status</dt><dd className="mt-1 font-semibold text-white/80">{diagnostics?.exact ? "Exact" : "Approximate / unsupported"}</dd></div>
      </dl>
      {activeCamera && !activeCamera.activeCameraMatchesBridge ? (
        <p className="border border-red-400/40 bg-red-400/10 p-2 text-xs font-bold text-red-200">Active renderer camera does not match calibrated camera.</p>
      ) : null}
      {activeCamera?.points.length ? (
        <details className="border-t border-white/10 pt-3 text-[11px] text-white/60">
          <summary className="cursor-pointer font-bold text-white/75">Active renderer landmark evidence</summary>
          <div className="mt-3 max-h-72 overflow-auto">
            <table className="w-full min-w-[42rem] text-left tabular-nums">
              <thead className="text-white/35"><tr><th className="pb-2 pr-3">Landmark</th><th className="pb-2 pr-3">OpenCV</th><th className="pb-2 pr-3">Bridge</th><th className="pb-2 pr-3">Active camera</th><th className="pb-2 pr-3">CV-bridge</th><th className="pb-2">Bridge-active</th></tr></thead>
              <tbody>{activeCamera.points.map((point) => (
                <tr key={point.semanticId} className="border-t border-white/5">
                  <td className="py-1.5 pr-3 font-semibold text-white/75">{point.semanticId}</td>
                  <td className="py-1.5 pr-3">{pixel(point.openCvPixel)}</td>
                  <td className="py-1.5 pr-3">{pixel(point.bridgePixel)}</td>
                  <td className="py-1.5 pr-3">{pixel(point.activeCameraPixel)}</td>
                  <td className="py-1.5 pr-3">{metric(point.openCvToBridgeError, " px")}</td>
                  <td className="py-1.5">{metric(point.bridgeToActiveCameraError, " px")}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </details>
      ) : null}
      {warnings.length ? (
        <ul className="space-y-1 border-t border-white/10 pt-3 text-[11px] leading-5 text-[#ffd998]">
          {warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </div>
  );
}
