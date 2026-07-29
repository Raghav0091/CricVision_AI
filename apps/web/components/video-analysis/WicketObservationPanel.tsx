"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import {
  getWicketObservations,
  runWicketObservations,
  type RealWicketObservation,
  type WicketObservationLandmark,
  type WicketObservationResult
} from "@/lib/api";


function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ");
}


function landmarkCounts(observation?: RealWicketObservation | null) {
  const landmarks = [
    ...(observation?.coarse_landmarks ?? []),
    ...(observation?.detailed_landmarks ?? [])
  ];
  return {
    available: landmarks.filter((item) => item.status === "AVAILABLE").length,
    high: landmarks.filter((item) => item.quality === "HIGH").length,
    medium: landmarks.filter((item) => item.quality === "MEDIUM").length,
    low: landmarks.filter((item) => item.quality === "LOW").length,
    unavailable: landmarks.filter((item) => item.status !== "AVAILABLE").length
  };
}


function WicketSummary({
  label,
  observation
}: {
  label: string;
  observation?: RealWicketObservation | null;
}) {
  const counts = landmarkCounts(observation);
  return (
    <section className="border-t border-white/10 pt-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-sm font-black text-white">{label}</h4>
        <span className="text-[10px] font-bold uppercase text-white/45">
          {observation ? readable(observation.region.stability) : "not found"}
        </span>
      </div>
      {observation ? (
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2 text-xs sm:grid-cols-4">
          <div>
            <dt className="text-white/35">Detector</dt>
            <dd className="mt-0.5 font-bold">{observation.region.detector_confidence.toFixed(2)}</dd>
          </div>
          <div>
            <dt className="text-white/35">Frame support</dt>
            <dd className="mt-0.5 font-bold">{observation.region.temporal_support}</dd>
          </div>
          <div>
            <dt className="text-white/35">Anchors</dt>
            <dd className="mt-0.5 font-bold">{counts.available}</dd>
          </div>
          <div>
            <dt className="text-white/35">Unavailable</dt>
            <dd className="mt-0.5 font-bold">{counts.unavailable}</dd>
          </div>
        </dl>
      ) : (
        <p className="mt-2 text-xs leading-5 text-white/45">
          No temporally supported region was observed. No replacement geometry was fabricated.
        </p>
      )}
      {observation && (
        <div className="mt-2 flex flex-wrap gap-2 text-[10px] font-bold uppercase">
          <span className="text-lime">{counts.high} high</span>
          <span className="text-[#ffd16a]">{counts.medium} medium</span>
          <span className="text-[#ff9e72]">{counts.low} low</span>
          <span className="text-white/35">
            detailed: {readable(observation.detailed_landmarks_status)}
          </span>
        </div>
      )}
    </section>
  );
}


export function WicketObservationPanel({ analysisId }: { analysisId: string }) {
  const [result, setResult] = useState<WicketObservationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    void getWicketObservations(analysisId)
      .then((stored) => {
        if (!cancelled) setResult(stored);
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Wicket observations are unavailable.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId]);

  const counts = useMemo(() => {
    const all: WicketObservationLandmark[] = [
      ...(result?.near_wicket?.coarse_landmarks ?? []),
      ...(result?.near_wicket?.detailed_landmarks ?? []),
      ...(result?.far_wicket?.coarse_landmarks ?? []),
      ...(result?.far_wicket?.detailed_landmarks ?? [])
    ];
    return {
      primary: all.filter((item) => item.registration_role === "PRIMARY_ANCHOR").length,
      secondary: all.filter((item) => item.registration_role === "SECONDARY_ANCHOR").length,
      validation: all.filter((item) => item.registration_role === "VALIDATION_ONLY").length
    };
  }, [result]);

  async function run() {
    setRunning(true);
    setError(null);
    try {
      setResult(await runWicketObservations(analysisId));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Wicket observation failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="border-t border-white/10 pt-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-black">Real Wicket Observations V1</h3>
          <p className="mt-1 text-xs leading-5 text-[#ffdc9a]">
            Real wicket observations only — camera registration has not yet been performed.
          </p>
        </div>
        <Button disabled={running} onClick={() => void run()}>
          {running ? "Observing…" : result ? "Run Again" : "Run Observation"}
        </Button>
      </div>

      {loading && <p className="mt-3 text-sm text-white/40">Checking stored observations…</p>}
      {error && (
        <p className="mt-3 border border-signal/30 bg-signal/10 px-3 py-2 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}

      {result && (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-white/10 py-3 text-xs">
            <span className="font-black uppercase text-white">{readable(result.status)}</span>
            <span className="text-white/45">
              Setup frame {result.setup_frame?.frame_index ?? "unavailable"}
            </span>
            <span className="text-white/45">
              Sampled {result.diagnostics.sampled_frame_ids.length} frames
            </span>
            <span className="text-lime">{counts.primary} primary</span>
            <span className="text-[#ffd16a]">{counts.secondary} secondary</span>
            <span className="text-[#ff9e72]">{counts.validation} validation</span>
          </div>

          {result.diagnostics.landmark_overlay_url && (
            <div
              className="relative mt-4 w-full overflow-hidden border border-white/10 bg-[#050a08]"
              style={{
                aspectRatio: result.setup_frame
                  ? `${result.setup_frame.image_width} / ${result.setup_frame.image_height}`
                  : "16 / 9"
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                alt="Real wicket detector regions and evidence-backed landmarks"
                className="h-full w-full object-contain"
                src={result.diagnostics.landmark_overlay_url}
              />
            </div>
          )}

          <div className="mt-3 flex flex-wrap gap-3 text-[10px] font-bold uppercase">
            <span className="text-white/55">Gray: raw detector</span>
            <span className="text-[#37d2ff]">Cyan: near consensus</span>
            <span className="text-[#ffbe46]">Amber: far consensus</span>
            <span className="text-lime">Green: primary anchor</span>
            <span className="text-[#ff9e72]">Orange: validation only</span>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <WicketSummary label="Near wicket candidate" observation={result.near_wicket} />
            <WicketSummary label="Far wicket candidate" observation={result.far_wicket} />
          </div>

          <section className="mt-4 border-t border-white/10 pt-3">
            <h4 className="text-sm font-black">End assignment remains unresolved</h4>
            <div className="mt-2 grid gap-2 text-xs sm:grid-cols-2">
              {result.assignment_hypotheses.map((hypothesis) => (
                <p key={hypothesis.hypothesis_id} className="leading-5 text-white/55">
                  <strong className="text-white">Hypothesis {hypothesis.hypothesis_id}:</strong>{" "}
                  near = {hypothesis.near_semantic_end}, far = {hypothesis.far_semantic_end}
                </p>
              ))}
            </div>
          </section>
        </>
      )}
    </section>
  );
}
