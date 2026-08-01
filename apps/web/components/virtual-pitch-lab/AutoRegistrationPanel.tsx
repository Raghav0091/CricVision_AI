import { Button } from "@/components/ui/Button";
import type {
  CameraSetupPreset,
  PresetAutoRegistrationResult
} from "@/lib/api";


type AutoRegistrationPanelProps = {
  analysisId: string;
  presetId: string;
  presets: CameraSetupPreset[];
  result: PresetAutoRegistrationResult | null;
  busy: boolean;
  loadingPresets: boolean;
  error: string | null;
  onAnalysisIdChange: (value: string) => void;
  onPresetIdChange: (value: string) => void;
  onRun: () => void;
  onClear: () => void;
  onOpenAdvanced: () => void;
};


const READY_STATUSES = new Set(["AUTO_REGISTRATION_READY", "VISUAL_OVERLAY_READY"]);


function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}


function metric(value: number | null | undefined, suffix = ""): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}${suffix}` : "Unavailable";
}


function ResultMetric({ label, value }: { label: string; value: string }) {
  return <div className="border-l-2 border-white/10 pl-3"><dt className="text-[10px] font-bold uppercase text-white/35">{label}</dt><dd className="mt-1 text-sm font-semibold text-white/80">{value}</dd></div>;
}


export function AutoRegistrationPanel({
  analysisId,
  presetId,
  presets,
  result,
  busy,
  loadingPresets,
  error,
  onAnalysisIdChange,
  onPresetIdChange,
  onRun,
  onClear,
  onOpenAdvanced
}: AutoRegistrationPanelProps) {
  const ready = Boolean(result && READY_STATUSES.has(result.status));
  const needsAssistance = result?.status === "NEEDS_ASSISTANCE"
    || result?.status === "PRESET_INCOMPATIBLE"
    || result?.status === "INSUFFICIENT_WICKETS"
    || result?.status === "INSUFFICIENT_EVIDENCE"
    || result?.status === "FAILED";
  const timings: PresetAutoRegistrationResult["stage_timings"] | null = result?.stage_timings ?? null;
  const fitted = Object.entries(result?.fitted_parameters ?? {}).filter(([, value]) => value !== null);
  const completedStages = result ? [
    "Preparing setup",
    result.detection_reused ? "Loading wicket observations" : "Detecting wickets",
    ...(result.candidates_attempted.length > 0 ? ["Fitting camera"] : []),
    ...(result.temporal_metrics ? ["Checking supporting frames"] : []),
    ...(Array.isArray(result.physical_checks) && result.physical_checks.length > 0 ? ["Validating geometry"] : []),
    ...(ready ? ["Setup complete"] : [])
  ] : [];

  return (
    <section className="border border-white/10 bg-white/[0.025]">
      <div className="border-b border-white/10 p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div><p className="text-xs font-black uppercase text-lime">One-click auto registration</p><p className="mt-1 text-xs text-white/40">Constrained rear-wicket setup, development only</p></div>
          {result && <span className={`border px-2.5 py-1 text-[10px] font-black uppercase ${ready ? "border-lime/35 bg-lime/10 text-lime" : needsAssistance ? "border-[#ffe761]/35 bg-[#ffe761]/10 text-[#ffe761]" : "border-white/15 text-white/55"}`}>{readable(result.status)}</span>}
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(14rem,0.65fr)_auto]">
          <label className="text-xs text-white/55">Analysis ID<input aria-label="Auto registration analysis ID" className="mt-2 w-full rounded border border-white/15 bg-[#0b1510] px-3 py-2.5 text-sm text-white" value={analysisId} onChange={(event) => onAnalysisIdChange(event.target.value)} /></label>
          <label className="text-xs text-white/55">Setup preset<select aria-label="Auto registration setup preset" className="mt-2 w-full rounded border border-white/15 bg-[#0b1510] px-3 py-2.5 text-sm font-semibold text-white" disabled={busy || loadingPresets} value={presetId} onChange={(event) => onPresetIdChange(event.target.value)}>{loadingPresets ? <option value="">Loading presets...</option> : presets.map((preset) => <option key={preset.preset_id} value={preset.preset_id}>{preset.preset_name}</option>)}</select></label>
          <div className="flex items-end gap-2"><Button className="min-h-10 whitespace-nowrap px-4 text-xs" disabled={busy || !analysisId.trim() || !presetId} onClick={onRun}>{busy ? "Aligning..." : "Auto Detect and Align"}</Button><Button className="min-h-10 px-3 text-xs" disabled={busy || !result} variant="secondary" onClick={onClear}>Clear Result</Button></div>
        </div>
      </div>

      {busy && <div aria-live="polite" className="border-b border-white/10 px-5 py-4"><p className="text-sm font-bold text-white">Preparing setup...</p><p className="mt-1 text-xs text-white/45">Requesting persisted wicket evidence and bounded fitting. Detailed stages will appear from the completed backend result.</p></div>}
      {error && <div role="alert" className="border-b border-[#ff6b6b]/20 bg-[#ff6b6b]/5 px-5 py-4 text-sm text-[#ff9b9b]">{error}</div>}

      {result && !busy && <div className="space-y-5 p-4 sm:p-5">
        <div>
          <p className="text-base font-black text-white">{ready ? "Setup complete" : needsAssistance ? "Automatic alignment needs assistance" : readable(result.status)}</p>
          <p className="mt-1 text-sm leading-6 text-white/50">{ready ? "The automatic camera is ready for visual overlay inspection. This is not an accepted production calibration." : "Review the evidence below before using advanced calibration."}</p>
        </div>

        <div aria-label="Completed automatic registration stages" className="flex flex-wrap gap-2">{completedStages.map((stage) => <span key={stage} className="border border-white/10 bg-black/15 px-2.5 py-1.5 text-[11px] font-semibold text-white/55">{stage}</span>)}</div>

        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <ResultMetric label="Preset" value={result.preset?.preset_name ?? presetId} />
          <ResultMetric label="Compatibility" value={readable(result.preset_compatibility?.status ?? "unknown")} />
          <ResultMetric label="Fit RMSE" value={metric(result.anchor_metrics?.reprojection_rmse_px, " px")} />
          <ResultMetric label="Median anchor error" value={metric(result.anchor_metrics?.median_reprojection_error_px, " px")} />
          <ResultMetric label="Temporal stability" value={metric(result.temporal_metrics?.temporal_stability_score)} />
          <ResultMetric label="Supporting frames" value={`${result.temporal_metrics?.successful_frame_count ?? 0} / ${result.temporal_metrics?.frame_count ?? result.supporting_frames?.length ?? 0}`} />
          <ResultMetric label="Uncertainty" value={result.uncertainty ? (result.uncertainty.stable ? "Stable" : "Needs review") : "Unavailable"} />
          <ResultMetric label="Pitch movement" value={metric(result.uncertainty?.projected_pitch_corner_movement_px, " px")} />
          <ResultMetric label="Near / far IoU" value={`${metric(result.envelope_metrics?.near_wicket_iou)} / ${metric(result.envelope_metrics?.far_wicket_iou)}`} />
          <ResultMetric label="Ambiguity" value={metric(result.ambiguity?.score)} />
          <ResultMetric label="Wicket evidence" value={result.detection_reused ? "Persisted observations reused" : "Detector rerun reported"} />
          <ResultMetric label="Classification" value={readable(result.geometric_classification)} />
        </dl>

        {fitted.length > 0 && <div><p className="text-[10px] font-bold uppercase text-white/35">Resulting camera</p><dl className="mt-2 grid gap-x-5 gap-y-2 text-xs sm:grid-cols-2 lg:grid-cols-3">{fitted.map(([key, value]) => <div key={key} className="flex justify-between gap-3 border-b border-white/5 py-1"><dt className="text-white/40">{readable(key)}</dt><dd className="tabular-nums text-white/75">{typeof value === "number" ? value.toFixed(3) : value}</dd></div>)}</dl></div>}

        {timings && <div><p className="text-[10px] font-bold uppercase text-white/35">Measured stage timings</p><div className="mt-2 flex flex-wrap gap-2">{Object.entries(timings).map(([stage, value]) => <span key={stage} className="border border-white/10 bg-black/15 px-2.5 py-1.5 text-xs text-white/55">{readable(stage)}: {metric(value, " ms")}</span>)}</div></div>}

        {(result.warnings.length > 0 || result.failure_reasons.length > 0 || result.preset_compatibility.reasons.length > 0) && <div className="border-t border-white/10 pt-4 text-xs leading-5 text-white/50">{[...result.failure_reasons, ...result.preset_compatibility.reasons.map((reason) => reason.message), ...result.warnings].map((item, index) => <p key={`${index}-${item}`}>{item}</p>)}</div>}

        {needsAssistance && <div className="flex flex-wrap gap-2 border-t border-white/10 pt-4"><Button className="text-xs" disabled={busy} onClick={onRun}>Retry</Button>{result.manual_assistance_available && <Button className="text-xs" variant="secondary" onClick={onOpenAdvanced}>Open Advanced Calibration</Button>}</div>}
      </div>}
    </section>
  );
}
