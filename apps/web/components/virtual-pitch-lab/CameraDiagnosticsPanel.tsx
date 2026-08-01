import type { CameraBridgeDiagnostics, LabCameraBridgeInput } from "./types";


function metric(value?: number | null, suffix = ""): string {
  return value == null || !Number.isFinite(value) ? "Unavailable" : `${value.toFixed(3)}${suffix}`;
}


export function CameraDiagnosticsPanel({
  camera,
  diagnostics
}: {
  camera: LabCameraBridgeInput | null;
  diagnostics?: CameraBridgeDiagnostics | null;
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
        <div><dt className="text-white/35">Classification</dt><dd className="mt-1 font-semibold text-white/80">{camera.classification}</dd></div>
        <div><dt className="text-white/35">Native frame</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{camera.image_width} x {camera.image_height}</dd></div>
        <div><dt className="text-white/35">Bridge RMSE</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{metric(diagnostics?.rmse_px, " px")}</dd></div>
        <div><dt className="text-white/35">Maximum error</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{metric(diagnostics?.maximum_error_px, " px")}</dd></div>
        <div><dt className="text-white/35">Landmarks</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{diagnostics ? `${diagnostics.valid_point_count}/${diagnostics.point_count}` : "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Invalid points</dt><dd className="mt-1 font-semibold tabular-nums text-white/80">{diagnostics?.invalid_point_count ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Distortion</dt><dd className="mt-1 break-words font-semibold text-white/80">{diagnostics?.distortion_mode ?? "Unavailable"}</dd></div>
        <div><dt className="text-white/35">Bridge status</dt><dd className="mt-1 font-semibold text-white/80">{diagnostics?.exact ? "Exact" : "Approximate / unsupported"}</dd></div>
      </dl>
      {warnings.length ? (
        <ul className="space-y-1 border-t border-white/10 pt-3 text-[11px] leading-5 text-[#ffd998]">
          {warnings.map((warning) => <li key={warning}>{warning}</li>)}
        </ul>
      ) : null}
    </div>
  );
}
