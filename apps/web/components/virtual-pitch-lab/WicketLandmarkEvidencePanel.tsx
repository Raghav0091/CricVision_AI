"use client";
/* eslint-disable @next/next/no-img-element -- native diagnostic pixels must not be transformed by Next image optimisation. */

import { useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import {
  clearWicketLandmarkEvidence,
  getPresetAutoRegistration,
  getWicketLandmarkEvidence,
  runPresetAutoRegistration,
  runWicketLandmarkEvidence,
  wicketLandmarkMediaUrl,
  type PresetAutoRegistrationResult,
  type WicketLandmarkEvidenceResult,
  type WicketLandmarkMedia,
  type WicketLandmarkSet
} from "@/lib/api";


type EvidencePanelProps = {
  analysisId: string;
  presetId: string;
  onAnalysisIdChange: (value: string) => void;
  onImprovedRegistration: (result: PresetAutoRegistrationResult) => void;
};

type ViewKey = keyof WicketLandmarkMedia;
type RegistrationPair = { legacy: PresetAutoRegistrationResult; improved: PresetAutoRegistrationResult };

const VIEW_OPTIONS: Array<{ key: ViewKey; label: string }> = [
  { key: "native_roi_url", label: "Native ROI" },
  { key: "temporal_consensus_url", label: "Temporal consensus" },
  { key: "raw_line_candidates_url", label: "Raw lines" },
  { key: "accepted_axes_url", label: "Accepted axes" },
  { key: "rejected_axes_url", label: "Rejected axes" },
  { key: "endpoints_url", label: "Endpoints" },
  { key: "uncertainty_url", label: "Uncertainty" },
  { key: "optional_scene_lines_url", label: "Scene lines" }
];

const DEFAULT_VIEWS = new Set<ViewKey>([
  "native_roi_url",
  "temporal_consensus_url",
  "accepted_axes_url",
  "endpoints_url",
  "uncertainty_url"
]);


function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}


function number(value: number | null | undefined, digits = 2, suffix = ""): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "Unavailable";
}


function evidenceCounts(wicket: WicketLandmarkSet | null) {
  const summary = wicket?.evidence_completeness;
  const points = wicket?.points ?? [];
  const axes = wicket?.axes ?? [];
  return {
    axes: summary?.detailed_axis_count ?? axes.filter((axis) => axis.status === "AVAILABLE").length,
    top: summary?.top_point_count ?? points.filter((point) => point.semantic_id.includes("top")).length,
    base: summary?.base_point_count ?? points.filter((point) => point.semantic_id.includes("base")).length,
    constraints: summary?.independent_constraint_count ?? 0,
    confidence: summary?.mean_confidence ?? wicket?.confidence,
    uncertainty: summary?.median_uncertainty_px ?? wicket?.uncertainty_px,
    grade: summary?.evidence_grade ?? "INSUFFICIENT"
  };
}


function mediaFor(wicket: WicketLandmarkSet | null, key: ViewKey): string | null {
  if (!wicket) return null;
  const debug = wicket.debug_media;
  const backendMedia = key === "native_roi_url"
    ? debug?.native_roi_image_url
    : key === "temporal_consensus_url"
      ? debug?.temporal_consensus_image_url
      : ["accepted_axes_url", "endpoints_url", "uncertainty_url"].includes(key)
        ? debug?.accepted_evidence_overlay_url
        : null;
  const fallback = key === "native_roi_url"
    ? wicket.native_roi?.image_url
    : key === "temporal_consensus_url" ? wicket.temporal_consensus_image_url : null;
  return wicketLandmarkMediaUrl(backendMedia ?? debug?.[key] ?? fallback);
}


function Metric({ label, value }: { label: string; value: string }) {
  return <div className="border-l-2 border-white/10 pl-3"><dt className="text-[10px] font-bold uppercase text-white/35">{label}</dt><dd className="mt-1 break-words text-sm font-semibold text-white/80">{value}</dd></div>;
}


function EvidenceSummary({ title, wicket }: { title: string; wicket: WicketLandmarkSet | null }) {
  const counts = evidenceCounts(wicket);
  const box = wicket?.native_roi?.box;
  return (
    <section className="min-w-0 border border-white/10 bg-black/15 p-3">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-black text-white">{title}</h3>
        <span className="border border-white/15 px-2 py-1 text-[10px] font-black text-white/65">{readable(counts.grade)}</span>
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
        <Metric label="Accepted axes" value={String(counts.axes)} />
        <Metric label="Top / base" value={`${counts.top} / ${counts.base}`} />
        <Metric label="Independent" value={String(counts.constraints)} />
        <Metric label="Confidence" value={number(counts.confidence)} />
        <Metric label="Uncertainty" value={number(counts.uncertainty, 2, " px")} />
        <Metric label="Alignment" value={number(wicket?.alignment_quality)} />
        <Metric label="Native ROI" value={box ? `${Math.round(box.width)} x ${Math.round(box.height)} at ${Math.round(box.x)}, ${Math.round(box.y)}` : "Unavailable"} />
      </dl>
      {!wicket && <p className="mt-3 text-xs text-white/40">No evidence was returned for this wicket.</p>}
    </section>
  );
}


function comparisonValue(result: PresetAutoRegistrationResult, key: "rmse" | "near" | "far" | "temporal" | "ambiguity") {
  if (key === "rmse") return result.anchor_metrics?.reprojection_rmse_px;
  if (key === "near") return result.envelope_metrics?.near_wicket_iou;
  if (key === "far") return result.envelope_metrics?.far_wicket_iou;
  if (key === "temporal") return result.temporal_metrics?.temporal_stability_score;
  return result.ambiguity?.score;
}


function RegistrationComparison({ pair }: { pair: RegistrationPair }) {
  const rows: Array<{ key: "rmse" | "near" | "far" | "temporal" | "ambiguity"; label: string; suffix?: string }> = [
    { key: "rmse", label: "Anchor RMSE", suffix: " px" },
    { key: "near", label: "Near IoU" },
    { key: "far", label: "Far IoU" },
    { key: "temporal", label: "Temporal" },
    { key: "ambiguity", label: "Ambiguity" }
  ];
  return (
    <div className="overflow-x-auto border border-white/10">
      <table className="w-full min-w-[34rem] text-left text-xs">
        <thead className="bg-white/[0.04] text-[10px] uppercase text-white/40"><tr><th className="p-3">Measure</th><th className="p-3">Legacy coarse box</th><th className="p-3">Improved landmarks</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.key} className="border-t border-white/10"><th className="p-3 font-semibold text-white/55">{row.label}</th><td className="p-3 tabular-nums text-white/70">{number(comparisonValue(pair.legacy, row.key), 3, row.suffix)}</td><td className="p-3 tabular-nums text-white/90">{number(comparisonValue(pair.improved, row.key), 3, row.suffix)}</td></tr>)}</tbody>
        <tfoot><tr className="border-t border-white/10"><th className="p-3 text-white/55">Status</th><td className="p-3 font-semibold text-white/70">{readable(pair.legacy.status)}</td><td className="p-3 font-semibold text-white/90">{readable(pair.improved.status)}</td></tr></tfoot>
      </table>
    </div>
  );
}


export function WicketLandmarkEvidencePanel({ analysisId, presetId, onAnalysisIdChange, onImprovedRegistration }: EvidencePanelProps) {
  const [evidence, setEvidence] = useState<WicketLandmarkEvidenceResult | null>(null);
  const [busyAction, setBusyAction] = useState<"load" | "run" | "clear" | "compare" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [views, setViews] = useState<Set<ViewKey>>(() => new Set(DEFAULT_VIEWS));
  const [comparison, setComparison] = useState<RegistrationPair | null>(null);

  const selectedFrames = useMemo(() => evidence?.supporting_frames.filter((frame) => typeof frame === "number" || frame.selected !== false).length ?? 0, [evidence]);
  const consideredFrames = evidence?.frame_selection?.frames_considered ?? evidence?.supporting_frames.length ?? 0;
  const constraints = evidence?.extraction_diagnostics?.independent_constraint_count
    ?? evidenceCounts(evidence?.near_wicket ?? null).constraints + evidenceCounts(evidence?.far_wicket ?? null).constraints;
  const mediaCount = evidence ? ([evidence.near_wicket, evidence.far_wicket] as const).reduce(
    (count, wicket) => count + VIEW_OPTIONS.filter((option) => views.has(option.key) && mediaFor(wicket, option.key)).length,
    0
  ) : 0;

  async function perform(action: Exclude<typeof busyAction, null>, operation: () => Promise<void>) {
    setBusyAction(action);
    setError(null);
    setMessage(null);
    try {
      await operation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The landmark evidence request failed.");
    } finally {
      setBusyAction(null);
    }
  }

  function validId(): string | null {
    const id = analysisId.trim();
    if (!id) setError("Enter an analysis ID first.");
    return id || null;
  }

  function toggleView(key: ViewKey) {
    setViews((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }

  async function compareRegistration() {
    const id = validId();
    if (!id || !presetId) return;
    await perform("compare", async () => {
      const common = { preset_id: presetId, reuse_existing_observations: true, force_redetect: false, development_diagnostics: true } as const;
      const legacy = await runPresetAutoRegistration(id, common);
      const extracted = await runWicketLandmarkEvidence(id, {
        reuse_existing_observations: true,
        force_redetect: false,
        include_optional_scene_evidence: false,
        rerun_auto_registration: true,
        write_debug_media: true,
        preset_id: presetId
      });
      const improved = await getPresetAutoRegistration(id);
      if (!improved) throw new Error("Improved registration did not produce a saved result.");
      setEvidence(extracted);
      setComparison({ legacy, improved });
      onImprovedRegistration(improved);
      setMessage("Legacy and improved modes reran through the same registration endpoint.");
    });
  }

  return (
    <details className="border border-white/10 bg-white/[0.02]" open>
      <summary className="cursor-pointer list-none p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-xs font-black uppercase text-[#5ee7ff]">Wicket Landmark Evidence</p><p className="mt-1 text-xs text-white/40">Native-frame, multi-frame developer diagnostics</p></div><span className="text-xs font-bold text-white/45">Collapse / expand</span></div>
      </summary>
      <div className="border-t border-white/10 p-4 sm:p-5">
        <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto]">
          <label className="min-w-0 text-xs text-white/55">Analysis ID<input aria-label="Wicket evidence analysis ID" className="mt-2 w-full rounded border border-white/15 bg-[#0b1510] px-3 py-2.5 text-sm text-white" value={analysisId} onChange={(event) => { onAnalysisIdChange(event.target.value); setEvidence(null); setComparison(null); }} /></label>
          <div className="flex flex-wrap items-end gap-2">
            <Button className="min-h-10 px-3 text-xs" disabled={Boolean(busyAction) || !analysisId.trim()} variant="secondary" onClick={() => { const id = validId(); if (id) void perform("load", async () => { const result = await getWicketLandmarkEvidence(id); setEvidence(result); setMessage(result ? "Saved evidence loaded." : "No saved landmark evidence exists for this analysis."); }); }}>Load</Button>
            <Button className="min-h-10 px-3 text-xs" disabled={Boolean(busyAction) || !analysisId.trim()} onClick={() => { const id = validId(); if (id) void perform("run", async () => { setEvidence(await runWicketLandmarkEvidence(id, { write_debug_media: true })); setMessage("Landmark extraction completed."); }); }}>{busyAction === "run" ? "Extracting..." : "Run extraction"}</Button>
            <Button className="min-h-10 px-3 text-xs" disabled={Boolean(busyAction) || !evidence} variant="danger" onClick={() => { const id = validId(); if (id) void perform("clear", async () => { await clearWicketLandmarkEvidence(id); setEvidence(null); setComparison(null); setMessage("Saved landmark evidence cleared."); }); }}>Clear</Button>
          </div>
        </div>

        {(message || error) && <p aria-live="polite" className={`mt-3 border px-3 py-2 text-xs ${error ? "border-[#ff6b6b]/25 bg-[#ff6b6b]/5 text-[#ff9b9b]" : "border-white/10 bg-black/15 text-white/55"}`}>{error ?? message}</p>}

        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
          {VIEW_OPTIONS.map((option) => <label key={option.key} className="flex min-h-10 cursor-pointer items-center gap-2 border border-white/10 bg-black/15 px-3 py-2 text-xs text-white/60"><input type="checkbox" className="h-4 w-4 accent-lime" checked={views.has(option.key)} onChange={() => toggleView(option.key)} />{option.label}</label>)}
        </div>

        {!evidence && <div className="mt-4 border border-dashed border-white/15 p-5 text-sm text-white/40">Load or run evidence to inspect native ROI crops, temporal consensus and extraction quality. Missing evidence stays unavailable.</div>}

        {evidence && <div className="mt-5 space-y-5">
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            <Metric label="Version" value={evidence.wicket_landmark_evidence_version} />
            <Metric label="Status" value={readable(evidence.status ?? "unknown")} />
            <Metric label="Native dimensions" value={`${evidence.native_image_width} x ${evidence.native_image_height}`} />
            <Metric label="Frames selected" value={`${selectedFrames} / ${consideredFrames}`} />
            <Metric label="Independent constraints" value={String(constraints)} />
            <Metric label="Median normalized residual" value={number(evidence.temporal_alignment?.median_normalized_residual, 3)} />
            <Metric label="Extraction" value={number(evidence.extraction_diagnostics?.landmark_extraction_consensus_ms, 1, " ms")} />
          </dl>

          <div className="grid gap-3 md:grid-cols-2"><EvidenceSummary title="Near wicket" wicket={evidence.near_wicket} /><EvidenceSummary title="Far wicket" wicket={evidence.far_wicket} /></div>

          <div className="grid gap-4 lg:grid-cols-2">
            {(["near_wicket", "far_wicket"] as const).flatMap((role) => VIEW_OPTIONS.filter((option) => views.has(option.key)).map((option) => {
              const url = mediaFor(evidence[role], option.key);
              if (!url) return null;
              return <figure key={`${role}-${option.key}`} className="min-w-0 overflow-hidden border border-white/10 bg-black/30"><figcaption className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2 text-xs font-bold text-white/60"><span>{role === "near_wicket" ? "Near" : "Far"}: {option.label}</span><span className="text-[10px] text-white/35">Native API media</span></figcaption><img className="block h-auto max-h-[28rem] w-full object-contain" src={url} alt={`${role === "near_wicket" ? "Near" : "Far"} wicket ${option.label.toLowerCase()}`} /></figure>;
            }))}
          </div>

          {mediaCount === 0 && <div className="border border-dashed border-white/15 p-4 text-xs leading-5 text-white/45">Selected native debug media is unavailable from the API. Local crop paths are intentionally not rendered; numerical ROI, alignment and landmark evidence remain visible above.</div>}

          <div className="flex flex-wrap gap-x-5 gap-y-2 border border-white/10 bg-black/15 px-3 py-2 text-[11px] text-white/60">
            <span><b className="mr-1 text-lime">+ Axis</b> accepted</span><span><b className="mr-1 text-[#ff9b9b]">x Axis</b> rejected</span><span><b className="mr-1 text-[#5ee7ff]">◇ Point</b> endpoint</span><span><b className="mr-1 text-[#ffe761]">±</b> uncertainty</span>
          </div>

          {(evidence.warnings.length > 0 || evidence.failure_reasons.length > 0) && <div className="border border-[#ffe761]/20 bg-[#ffe761]/5 p-3 text-xs leading-5 text-[#ffeaa0]">{[...evidence.failure_reasons, ...evidence.warnings].map((warning, index) => <p key={`${index}-${warning}`}>{warning}</p>)}</div>}

          <div className="border-t border-white/10 pt-4">
            <div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-sm font-black text-white">Legacy versus improved registration</p><p className="mt-1 text-xs text-white/40">Runs the existing solver twice with explicit evidence modes; no calibration is accepted.</p></div><Button className="text-xs" disabled={Boolean(busyAction) || !presetId} variant="secondary" onClick={() => void compareRegistration()}>{busyAction === "compare" ? "Comparing..." : "Rerun comparison"}</Button></div>
            {comparison && <div className="mt-4"><RegistrationComparison pair={comparison} /></div>}
          </div>
        </div>}
      </div>
    </details>
  );
}
