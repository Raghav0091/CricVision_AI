"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import {
  getRealPitchRegistration,
  runRealPitchRegistration,
  type RealPitchProjection,
  type RealPitchRegistrationResult,
  type RegistrationCandidate
} from "@/lib/api";


function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ");
}


function score(value?: number | null): string {
  return value == null ? "Unavailable" : value.toFixed(3);
}


function lineStyle(category: string) {
  if (category === "centreline") {
    return { colour: "#b8f276", dash: "8 7", width: 2 };
  }
  if (category === "pitch_boundary") {
    return { colour: "#55d4f4", width: 2 };
  }
  if (category === "bowling_crease") {
    return { colour: "#ffd35f", width: 2.5 };
  }
  if (category === "popping_crease") {
    return { colour: "#ff9d5c", width: 2.5 };
  }
  return { colour: "#d8c7ff", width: 2 };
}


function RegistrationOverlay({
  result,
  projection,
  candidate,
  opacity,
  showOverlay,
  showBoxes,
  showObserved,
  showProjected,
  showRejected
}: {
  result: RealPitchRegistrationResult;
  projection: RealPitchProjection;
  candidate: RegistrationCandidate;
  opacity: number;
  showOverlay: boolean;
  showBoxes: boolean;
  showObserved: boolean;
  showProjected: boolean;
  showRejected: boolean;
}) {
  const camera = projection.source_camera;
  const surface = projection.projected_polygons.find(
    (item) => item.primitive_id === "pitch_surface"
  );
  const corridor = projection.projected_polygons.find(
    (item) => item.polygon_category === "lbw_corridor"
  );
  const polygonPoints = (polygon: typeof surface) => {
    if (!polygon?.projection_valid || polygon.pixel_vertices.some((item) => !item)) {
      return null;
    }
    return polygon.pixel_vertices.map((item) => `${item?.x},${item?.y}`).join(" ");
  };
  const residualById = new Map(
    candidate.reprojection_residuals.map((item) => [item.correspondence_id, item])
  );

  return (
    <svg
      aria-label="Real camera pose candidate projection"
      className="absolute inset-0 h-full w-full"
      viewBox={`0 0 ${camera.image_width} ${camera.image_height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      {showOverlay && (
        <g opacity={opacity}>
          {polygonPoints(surface) && (
            <polygon
              fill="#1d6b49"
              fillOpacity="0.16"
              points={polygonPoints(surface) ?? ""}
              stroke="#6ed4a1"
              strokeWidth="2"
            />
          )}
          {polygonPoints(corridor) && (
            <polygon
              fill="#ffd35f"
              fillOpacity="0.13"
              points={polygonPoints(corridor) ?? ""}
              stroke="#ffd35f"
              strokeDasharray="7 6"
              strokeWidth="2"
            />
          )}
          {projection.projected_line_segments.map((line) => {
            if (!line.projection_valid || !line.pixel_start || !line.pixel_end) return null;
            const style = lineStyle(line.line_category);
            return (
              <line
                key={line.primitive_id}
                x1={line.pixel_start.x}
                y1={line.pixel_start.y}
                x2={line.pixel_end.x}
                y2={line.pixel_end.y}
                stroke={style.colour}
                strokeDasharray={style.dash}
                strokeLinecap="round"
                strokeWidth={style.width}
              />
            );
          })}
          {projection.projected_stumps.map((stump) => {
            if (!stump.projection_valid || !stump.pixel_base || !stump.pixel_top) return null;
            return (
              <line
                key={stump.primitive_id}
                x1={stump.pixel_base.x}
                y1={stump.pixel_base.y}
                x2={stump.pixel_top.x}
                y2={stump.pixel_top.y}
                stroke="#fff8d4"
                strokeLinecap="round"
                strokeWidth={Math.max(2, (stump.projected_radius_px ?? 1) * 2)}
              />
            );
          })}
          {projection.projected_bails.map((bail) => {
            if (!bail.projection_valid || !bail.pixel_start || !bail.pixel_end) return null;
            return (
              <line
                key={bail.primitive_id}
                x1={bail.pixel_start.x}
                y1={bail.pixel_start.y}
                x2={bail.pixel_end.x}
                y2={bail.pixel_end.y}
                stroke="#ffd35f"
                strokeLinecap="round"
                strokeWidth="2.5"
              />
            );
          })}
        </g>
      )}
      {showBoxes && result.correspondences
        .filter((item) => item.mapping_type === "WICKET_ENVELOPE" && item.observed_bbox)
        .map((item) => (
          <rect
            key={item.correspondence_id}
            x={item.observed_bbox?.x}
            y={item.observed_bbox?.y}
            width={item.observed_bbox?.width}
            height={item.observed_bbox?.height}
            fill="none"
            stroke={item.observed_wicket_role === "near" ? "#37d2ff" : "#ffbe46"}
            strokeDasharray="7 5"
            strokeWidth="3"
          />
        ))}
      {result.correspondences.map((item) => {
        const residual = residualById.get(item.correspondence_id);
        if (!residual) return null;
        const colour = residual.inlier ? "#b8f276" : "#ff6961";
        return (
          <g key={item.correspondence_id}>
            {showObserved && (
              <circle
                cx={residual.observed_pixel.x}
                cy={residual.observed_pixel.y}
                fill={colour}
                r="4"
                stroke="#07110d"
                strokeWidth="1.5"
              />
            )}
            {showProjected && (
              <circle
                cx={residual.projected_pixel.x}
                cy={residual.projected_pixel.y}
                fill="none"
                r="5"
                stroke="#ffffff"
                strokeWidth="2"
              />
            )}
            {showObserved && showProjected && (
              <line
                x1={residual.observed_pixel.x}
                y1={residual.observed_pixel.y}
                x2={residual.projected_pixel.x}
                y2={residual.projected_pixel.y}
                stroke={colour}
                strokeWidth="1.5"
              />
            )}
          </g>
        );
      })}
      {showRejected && result.correspondences
        .filter((item) => item.status === "REJECTED" && item.observed_pixel)
        .map((item) => (
          <g key={item.correspondence_id}>
            <line
              x1={(item.observed_pixel?.x ?? 0) - 5}
              y1={(item.observed_pixel?.y ?? 0) - 5}
              x2={(item.observed_pixel?.x ?? 0) + 5}
              y2={(item.observed_pixel?.y ?? 0) + 5}
              stroke="#ff6961"
              strokeWidth="2"
            />
            <line
              x1={(item.observed_pixel?.x ?? 0) + 5}
              y1={(item.observed_pixel?.y ?? 0) - 5}
              x2={(item.observed_pixel?.x ?? 0) - 5}
              y2={(item.observed_pixel?.y ?? 0) + 5}
              stroke="#ff6961"
              strokeWidth="2"
            />
          </g>
        ))}
    </svg>
  );
}


function CandidateStats({ candidate }: { candidate: RegistrationCandidate }) {
  const position = candidate.camera_world_position;
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs sm:grid-cols-4">
      <div>
        <dt className="text-white/35">Classification</dt>
        <dd className="mt-1 font-bold uppercase">{readable(candidate.classification)}</dd>
      </div>
      <div>
        <dt className="text-white/35">Reprojection RMSE</dt>
        <dd className="mt-1 font-bold">
          {candidate.reprojection_rmse_px == null
            ? "Unavailable"
            : `${candidate.reprojection_rmse_px.toFixed(2)} px`}
        </dd>
      </div>
      <div>
        <dt className="text-white/35">Inliers / outliers</dt>
        <dd className="mt-1 font-bold">
          {candidate.inlier_correspondence_ids.length} / {candidate.outlier_correspondence_ids.length}
        </dd>
      </div>
      <div>
        <dt className="text-white/35">Focal source</dt>
        <dd className="mt-1 font-bold">{readable(candidate.intrinsics.source)}</dd>
      </div>
      <div>
        <dt className="text-white/35">Focal length</dt>
        <dd className="mt-1 font-bold">{candidate.intrinsics.focal_length_x_px.toFixed(1)} px</dd>
      </div>
      <div>
        <dt className="text-white/35">Camera position</dt>
        <dd className="mt-1 font-bold">
          {position ? position.map((item) => item.toFixed(2)).join(", ") : "Unavailable"}
        </dd>
      </div>
      <div>
        <dt className="text-white/35">Temporal stability</dt>
        <dd className="mt-1 font-bold">{score(candidate.temporal_validation?.stability_score)}</dd>
      </div>
      <div>
        <dt className="text-white/35">Independent scene</dt>
        <dd className="mt-1 font-bold">
          {score(candidate.independent_validation?.independent_scene_score)}
        </dd>
      </div>
      <div>
        <dt className="text-white/35">Position spread</dt>
        <dd className="mt-1 font-bold">
          {candidate.uncertainty?.camera_position_spread_m == null
            ? "Unavailable"
            : `${candidate.uncertainty.camera_position_spread_m.toFixed(3)} m`}
        </dd>
      </div>
      <div>
        <dt className="text-white/35">Overlay sensitivity</dt>
        <dd className="mt-1 font-bold">
          {candidate.uncertainty?.maximum_overlay_movement_px == null
            ? "Unavailable"
            : `${candidate.uncertainty.maximum_overlay_movement_px.toFixed(2)} px`}
        </dd>
      </div>
    </dl>
  );
}


export function RealPitchRegistrationPanel({ analysisId }: { analysisId: string }) {
  const [result, setResult] = useState<RealPitchRegistrationResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.8);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showObserved, setShowObserved] = useState(true);
  const [showProjected, setShowProjected] = useState(true);
  const [showRejected, setShowRejected] = useState(false);
  const [comparison, setComparison] = useState<"selected" | "competing">("selected");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    void getRealPitchRegistration(analysisId)
      .then((stored) => {
        if (!cancelled) setResult(stored);
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Registration is unavailable.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [analysisId]);

  const activeCandidate = comparison === "competing"
    ? result?.competing_candidate
    : result?.selected_candidate;
  const activeProjection = comparison === "competing"
    ? result?.competing_projected_pitch_geometry
    : result?.projected_pitch_geometry;
  const unavailable = useMemo(
    () => result?.correspondences.filter((item) => item.status === "UNAVAILABLE") ?? [],
    [result]
  );

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const next = await runRealPitchRegistration(analysisId);
      setResult(next);
      setComparison("selected");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Registration failed.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="border-t border-white/10 pt-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-black">Real Camera Pose Registration V1</h3>
          <p className="mt-1 text-xs leading-5 text-[#ffdc9a]">
            Real camera pose candidate - not yet accepted for metric analytics.
          </p>
        </div>
        <Button disabled={running} onClick={() => void run()}>
          {running ? "Solving..." : result ? "Run Again" : "Run Registration"}
        </Button>
      </div>

      {loading && <p className="mt-3 text-sm text-white/40">Checking stored candidates...</p>}
      {error && (
        <p className="mt-3 border border-signal/30 bg-signal/10 px-3 py-2 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}

      {result && (
        <>
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-white/10 py-3 text-xs">
            <strong className="uppercase">{readable(result.status)}</strong>
            <span className="text-white/45">{result.diagnostics.pose_candidate_count} pose candidates</span>
            <span className="text-white/45">{result.diagnostics.focal_candidate_count} focal hypotheses</span>
            <span className="text-white/45">ambiguity {result.ambiguity_score.toFixed(3)}</span>
            <span className="font-bold text-[#ffdc9a]">metrics locked</span>
          </div>

          {result.competing_candidate && result.competing_projected_pitch_geometry && (
            <div className="mt-3 inline-flex border border-white/10 p-1">
              {(["selected", "competing"] as const).map((item) => (
                <button
                  key={item}
                  type="button"
                  className={`px-3 py-1.5 text-xs font-bold ${
                    comparison === item ? "bg-lime text-[#07110d]" : "text-white/55"
                  }`}
                  onClick={() => setComparison(item)}
                >
                  {item === "selected" ? "Selected hypothesis" : "Competing hypothesis"}
                </button>
              ))}
            </div>
          )}

          {activeCandidate && activeProjection && result.diagnostics.setup_frame_image_url && (
            <>
              <div
                className="relative mt-4 w-full overflow-hidden border border-white/10 bg-[#050a08]"
                style={{
                  aspectRatio: `${activeProjection.source_camera.image_width} / ${activeProjection.source_camera.image_height}`
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  alt="Real setup frame"
                  className="absolute inset-0 h-full w-full object-contain"
                  src={result.diagnostics.setup_frame_image_url}
                />
                <RegistrationOverlay
                  result={result}
                  projection={activeProjection}
                  candidate={activeCandidate}
                  opacity={opacity}
                  showOverlay={showOverlay}
                  showBoxes={showBoxes}
                  showObserved={showObserved}
                  showProjected={showProjected}
                  showRejected={showRejected}
                />
              </div>

              <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-3">
                <label className="flex items-center gap-2">
                  <input checked={showOverlay} type="checkbox" onChange={(event) => setShowOverlay(event.target.checked)} />
                  Virtual pitch
                </label>
                <label className="flex items-center gap-2">
                  <input checked={showBoxes} type="checkbox" onChange={(event) => setShowBoxes(event.target.checked)} />
                  Raw wicket boxes
                </label>
                <label className="flex items-center gap-2">
                  <input checked={showObserved} type="checkbox" onChange={(event) => setShowObserved(event.target.checked)} />
                  Observed anchors
                </label>
                <label className="flex items-center gap-2">
                  <input checked={showProjected} type="checkbox" onChange={(event) => setShowProjected(event.target.checked)} />
                  Projected anchors
                </label>
                <label className="flex items-center gap-2">
                  <input checked={showRejected} type="checkbox" onChange={(event) => setShowRejected(event.target.checked)} />
                  Rejected correspondences
                </label>
                <label className="flex items-center gap-2">
                  <span>Opacity</span>
                  <input
                    aria-label="Registration overlay opacity"
                    className="min-w-0 flex-1 accent-lime"
                    max="1"
                    min="0.1"
                    step="0.05"
                    type="range"
                    value={opacity}
                    onChange={(event) => setOpacity(Number(event.target.value))}
                  />
                </label>
              </div>

              <section className="mt-4 border-t border-white/10 pt-3">
                <h4 className="mb-3 text-sm font-black">
                  Hypothesis {activeCandidate.assignment_hypothesis}: near = {activeCandidate.near_semantic_end},
                  {" "}far = {activeCandidate.far_semantic_end}
                </h4>
                <CandidateStats candidate={activeCandidate} />
              </section>
            </>
          )}

          {!result.attempted && (
            <section className="mt-4 border-t border-white/10 pt-3">
              <h4 className="text-sm font-black">Eligibility gate</h4>
              <ul className="mt-2 space-y-1 text-xs text-white/55">
                {result.failure_reasons.map((reason) => <li key={reason}>{readable(reason)}</li>)}
              </ul>
            </section>
          )}

          <details className="mt-4 border-t border-white/10 pt-3">
            <summary className="cursor-pointer text-xs font-bold text-white/55">
              Correspondence diagnostics ({unavailable.length} unavailable)
            </summary>
            <div className="mt-2 max-h-52 overflow-auto text-xs text-white/45">
              {result.correspondences.map((item) => (
                <p key={item.correspondence_id} className="border-b border-white/5 py-1.5">
                  <strong className="text-white/70">{item.correspondence_id}</strong>
                  {" · "}{readable(item.exactness)} · {readable(item.status)}
                  {item.rejection_reason ? ` · ${readable(item.rejection_reason)}` : ""}
                </p>
              ))}
            </div>
          </details>
        </>
      )}
    </section>
  );
}
