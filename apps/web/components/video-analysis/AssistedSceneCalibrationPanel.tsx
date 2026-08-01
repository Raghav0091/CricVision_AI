"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent
} from "react";

import { Button } from "@/components/ui/Button";
import {
  acceptSceneCalibration,
  clearSceneCalibrationOrientation,
  confirmSceneCalibrationOrientation,
  getSceneCalibration,
  refineSceneCalibration,
  rejectSceneCalibration,
  runSceneCalibration,
  saveSceneCalibrationAnchors,
  enableVisualSceneCalibration,
  type ImageLeftMapping,
  type RealPitchProjection,
  type RegistrationCandidate,
  type SceneCalibrationAnchor,
  type SceneCalibrationAnchorInput,
  type SceneCalibrationResult
} from "@/lib/api";
import {
  displayToVideoNative,
  type CalibrationViewTransform
} from "@/lib/calibrationCoordinates";

import { AnalysisMediaStage } from "./AnalysisMediaStage";


const REQUIRED_ANCHORS = [
  "near_left_base",
  "near_right_base",
  "near_top_center",
  "far_left_base",
  "far_right_base",
  "far_top_center"
];
const CREASE_ANCHORS = [
  "near_popping_crease_left",
  "near_popping_crease_right",
  "far_popping_crease_left",
  "far_popping_crease_right"
];
const SEMANTIC_SCENE_ANCHORS = [
  ...CREASE_ANCHORS,
  "pitch_left_edge_reference",
  "pitch_right_edge_reference"
];


function readable(value: string): string {
  return value.toLowerCase().replaceAll("_", " ");
}


function qualityLabel(result: SceneCalibrationResult): string {
  if (result.stage === "INSUFFICIENT_EVIDENCE") return "Insufficient evidence";
  if (result.stage === "FAILED") return "Failed";
  if (result.calibration_level === "METRIC_3D_READY") return "Metric 3D candidate";
  if (result.calibration_level === "GROUND_PLANE_READY") return "Ground-plane candidate";
  if (result.calibration_level === "VISUAL_ONLY") return "Visual only";
  return "Not calibrated";
}


function lineColour(category: string): string {
  if (category === "pitch_boundary") return "#55d4f4";
  if (category === "bowling_crease") return "#ffd35f";
  if (category === "popping_crease") return "#ff9d5c";
  if (category === "return_crease") return "#d8c7ff";
  return "#b8f276";
}


function PitchProjection({
  projection,
  opacity,
  showProjectedAnchors
}: {
  projection: RealPitchProjection;
  opacity: number;
  showProjectedAnchors: boolean;
}) {
  return (
    <g opacity={opacity} pointerEvents="none">
      {projection.projected_polygons.map((polygon) => {
        if (!polygon.projection_valid || polygon.pixel_vertices.some((point) => !point)) {
          return null;
        }
        const points = polygon.pixel_vertices.map((point) => `${point?.x},${point?.y}`).join(" ");
        return (
          <polygon
            key={polygon.primitive_id}
            points={points}
            fill={polygon.polygon_category === "lbw_corridor" ? "#ffd35f" : "#1d6b49"}
            fillOpacity={polygon.polygon_category === "lbw_corridor" ? 0.12 : 0.08}
            stroke={polygon.polygon_category === "lbw_corridor" ? "#ffd35f" : "#6ed4a1"}
            strokeWidth="2"
          />
        );
      })}
      {projection.projected_line_segments.map((line) => (
        line.projection_valid && line.pixel_start && line.pixel_end ? (
          <line
            key={line.primitive_id}
            x1={line.pixel_start.x}
            y1={line.pixel_start.y}
            x2={line.pixel_end.x}
            y2={line.pixel_end.y}
            stroke={lineColour(line.line_category)}
            strokeWidth={line.line_category.includes("crease") ? 2.5 : 2}
            strokeDasharray={line.line_category === "centreline" ? "8 7" : undefined}
          />
        ) : null
      ))}
      {[...projection.projected_stumps].map((stump) => (
        stump.projection_valid && stump.pixel_base && stump.pixel_top ? (
          <line
            key={stump.primitive_id}
            x1={stump.pixel_base.x}
            y1={stump.pixel_base.y}
            x2={stump.pixel_top.x}
            y2={stump.pixel_top.y}
            stroke="#fff8d4"
            strokeWidth="3"
            strokeLinecap="round"
          />
        ) : null
      ))}
      {projection.projected_bails.map((bail) => (
        bail.projection_valid && bail.pixel_start && bail.pixel_end ? (
          <line
            key={bail.primitive_id}
            x1={bail.pixel_start.x}
            y1={bail.pixel_start.y}
            x2={bail.pixel_end.x}
            y2={bail.pixel_end.y}
            stroke="#ffd35f"
            strokeWidth="2.5"
          />
        ) : null
      ))}
      {showProjectedAnchors && projection.projected_landmarks.map((landmark) => (
        landmark.projection_valid && landmark.pixel_point ? (
          <circle
            key={landmark.semantic_id}
            cx={landmark.pixel_point.x}
            cy={landmark.pixel_point.y}
            r="3"
            fill="#ffffff"
            stroke="#07110d"
            strokeWidth="1"
          />
        ) : null
      ))}
    </g>
  );
}


function anchorInput(anchor: SceneCalibrationAnchor): SceneCalibrationAnchorInput {
  return {
    semantic_id: anchor.semantic_id,
    video_point: anchor.video_point,
    source: anchor.source,
    used_for_refinement: anchor.used_for_refinement,
    used_for_validation: anchor.used_for_validation
  };
}


function mappingText(mapping?: string | null): string {
  if (mapping === "IMAGE_LEFT_IS_PITCH_LEFT" || mapping === "image_left_to_world_left") {
    return "Image left = pitch left";
  }
  if (mapping === "IMAGE_LEFT_IS_PITCH_RIGHT" || mapping === "image_left_to_world_right") {
    return "Image left = pitch right";
  }
  return "Unresolved";
}


function CandidateSummary({
  label,
  candidate,
  active
}: {
  label: string;
  candidate?: RegistrationCandidate | null;
  active: boolean;
}) {
  return (
    <div className={`border px-3 py-2 text-xs ${active ? "border-lime/50 bg-lime/[0.06]" : "border-white/10 bg-black/15"}`}>
      <p className="font-black uppercase">{label}</p>
      <p className="mt-1 text-white/55">{candidate?.candidate_id ?? "Unavailable"}</p>
      <p className="mt-1">{mappingText(candidate?.lateral_mapping)}</p>
      <p className="mt-1 text-white/45">
        score {candidate ? candidate.score.toFixed(3) : "n/a"} / rmse {candidate?.reprojection_rmse_px?.toFixed(2) ?? "n/a"} px
      </p>
    </div>
  );
}


export function AssistedSceneCalibrationPanel({
  analysisId,
  initialResult,
  onResult,
  onContinue
}: {
  analysisId: string;
  initialResult?: SceneCalibrationResult | null;
  onResult?: (result: SceneCalibrationResult) => void;
  onContinue?: () => void;
}) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ pointerId: number; semanticId: string } | null>(null);
  const [result, setResult] = useState<SceneCalibrationResult | null>(initialResult ?? null);
  const [anchors, setAnchors] = useState<SceneCalibrationAnchor[]>([]);
  const [history, setHistory] = useState<SceneCalibrationAnchor[][]>([]);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [placing, setPlacing] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showAnchors, setShowAnchors] = useState(true);
  const [showAutomatic, setShowAutomatic] = useState(false);
  const [showProjectedAnchors, setShowProjectedAnchors] = useState(false);
  const [opacity, setOpacity] = useState(0.78);
  const [view, setView] = useState<CalibrationViewTransform>({
    zoom: 1,
    panX: 0,
    panY: 0
  });
  const [candidateView, setCandidateView] = useState<"selected" | "competing">("selected");
  const [savePreset, setSavePreset] = useState(false);

  function adopt(next: SceneCalibrationResult) {
    setResult(next);
    setAnchors([...next.current_anchor_set, ...next.optional_crease_anchors]);
    onResult?.(next);
  }

  useEffect(() => {
    if (initialResult) {
      adopt(initialResult);
      return;
    }
    let cancelled = false;
    void getSceneCalibration(analysisId)
      .then((next) => {
        if (!cancelled) adopt(next);
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Calibration is unavailable.");
        }
      });
    return () => {
      cancelled = true;
    };
    // The analysis ID owns the calibration lifecycle.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisId]);

  const setup = result?.setup_frame;
  const projection = candidateView === "competing"
    ? result?.competing_projected_pitch_geometry ?? result?.projected_pitch_geometry
    : result?.projected_pitch_geometry;
  const canEdit = Boolean(setup && result?.selected_candidate);
  const candidateReady = Boolean(
    result?.validation?.all_required_checks_passed
    && ["GROUND_PLANE_READY", "METRIC_3D_READY"].includes(
      result.validation.eligible_level
    )
  );
  const terminal = Boolean(
    result && ![
      "NOT_STARTED",
      "DETECTING_WICKETS",
      "OBSERVING_WICKETS",
      "GENERATING_POSE"
    ].includes(result.stage)
  );
  const requiredUnavailable = useMemo(
    () => REQUIRED_ANCHORS.filter((id) => !anchors.find(
      (anchor) => anchor.semantic_id === id && anchor.video_point
    )),
    [anchors]
  );
  const orientationRequired = Boolean(
    result?.orientation_required || result?.stage === "ORIENTATION_REQUIRED"
  );

  async function perform(message: string, action: () => Promise<SceneCalibrationResult>) {
    setBusy(true);
    setBusyMessage(message);
    setError(null);
    try {
      adopt(await action());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Calibration action failed.");
    } finally {
      setBusy(false);
      setBusyMessage("");
    }
  }

  async function detect() {
    setHistory([]);
    setEditing(false);
    await perform("Detecting wickets and aligning the virtual pitch...", () =>
      runSceneCalibration(analysisId)
    );
  }

  async function confirmOrientation(mapping: ImageLeftMapping | "NOT_SURE") {
    if (!result) return;
    await perform("Applying pitch orientation evidence...", () =>
      confirmSceneCalibrationOrientation(
        analysisId,
        result.anchor_version,
        mapping,
        {
          cameraEnd: "unknown",
          createPreset: savePreset && mapping !== "NOT_SURE",
          presetName: "Fixed camera orientation",
          userConfirmedSameFixedSetup: savePreset && mapping !== "NOT_SURE"
        }
      )
    );
  }

  async function clearOrientation() {
    if (!result) return;
    await perform("Clearing pitch orientation...", () =>
      clearSceneCalibrationOrientation(analysisId, result.anchor_version)
    );
  }

  function updateLocalAnchor(semanticId: string, x: number, y: number) {
    setAnchors((current) => {
      const existing = current.find((item) => item.semantic_id === semanticId);
      const source = existing?.original_automatic_point
        ? "manually_adjusted"
        : "manually_added";
      const updated: SceneCalibrationAnchor = existing
        ? {
            ...existing,
            video_point: { x, y },
            source,
            valid: true,
            validation_messages: []
          }
        : {
            semantic_id: semanticId,
            kind: CREASE_ANCHORS.includes(semanticId)
              ? "crease"
              : semanticId.includes("pitch_")
                ? "pitch_edge"
                : "wicket",
            wicket_role: semanticId.startsWith("near") ? "near" : "far",
            video_point: { x, y },
            source,
            original_automatic_point: null,
            confidence: 0.72,
            uncertainty_px: 4,
            adjustment_distance_px: 0,
            frame_index: setup?.frame_index ?? 0,
            valid: true,
            used_for_refinement: true,
            used_for_validation: true,
            validation_messages: []
          };
      return [...current.filter((item) => item.semantic_id !== semanticId), updated];
    });
  }

  function nativePointFromEvent(event: ReactPointerEvent<Element>) {
    if (!setup || !stageRef.current) return null;
    const rect = stageRef.current.getBoundingClientRect();
    return displayToVideoNative(
      { x: event.clientX - rect.left, y: event.clientY - rect.top },
      { width: setup.image_width, height: setup.image_height },
      { width: rect.width, height: rect.height },
      view
    );
  }

  function startDrag(event: ReactPointerEvent<SVGCircleElement>, semanticId: string) {
    if (!editing || busy) return;
    event.preventDefault();
    event.stopPropagation();
    setHistory((current) => [...current, anchors]);
    dragRef.current = { pointerId: event.pointerId, semanticId };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveDrag(event: ReactPointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const point = nativePointFromEvent(event);
    if (point) updateLocalAnchor(drag.semanticId, point.x, point.y);
  }

  async function endDrag(event: ReactPointerEvent<SVGSVGElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !result) return;
    dragRef.current = null;
    const point = nativePointFromEvent(event);
    const existing = anchors.find((item) => item.semantic_id === drag.semanticId);
    if (!point || !existing) return;
    const changed: SceneCalibrationAnchor = {
      ...existing,
      video_point: point,
      source: existing.original_automatic_point
        ? "manually_adjusted"
        : "manually_added"
    };
    await perform("Saving anchor position...", () =>
      saveSceneCalibrationAnchors(analysisId, result.anchor_version, [anchorInput(changed)])
    );
  }

  async function placeAnchor(event: ReactPointerEvent<SVGSVGElement>) {
    if (!placing || !result) return;
    const point = nativePointFromEvent(event);
    if (!point) return;
    const semanticId = placing;
    setHistory((current) => [...current, anchors]);
    updateLocalAnchor(semanticId, point.x, point.y);
    setPlacing(null);
    await perform("Saving added anchor...", () =>
      saveSceneCalibrationAnchors(analysisId, result.anchor_version, [{
        semantic_id: semanticId,
        video_point: point,
        source: "manually_added",
        used_for_refinement: true,
        used_for_validation: true
      }])
    );
  }

  async function resetAnchor(semanticId: string) {
    if (!result) return;
    const anchor = anchors.find((item) => item.semantic_id === semanticId);
    if (!anchor?.original_automatic_point) return;
    setHistory((current) => [...current, anchors]);
    await perform("Resetting anchor...", () =>
      saveSceneCalibrationAnchors(analysisId, result.anchor_version, [{
        semantic_id: semanticId,
        video_point: anchor.original_automatic_point,
        source: "manually_adjusted",
        used_for_refinement: true,
        used_for_validation: true
      }])
    );
  }

  async function resetAll() {
    if (!result) return;
    const reset = anchors
      .filter((anchor) => anchor.original_automatic_point)
      .map((anchor) => ({
        semantic_id: anchor.semantic_id,
        video_point: anchor.original_automatic_point,
        source: "manually_adjusted" as const,
        used_for_refinement: anchor.used_for_refinement,
        used_for_validation: anchor.used_for_validation
      }));
    if (!reset.length) return;
    setHistory((current) => [...current, anchors]);
    await perform("Resetting automatic anchors...", () =>
      saveSceneCalibrationAnchors(analysisId, result.anchor_version, reset)
    );
  }

  async function removeOptionalAnchor(semanticId: string) {
    if (!result) return;
    const anchor = anchors.find((item) => item.semantic_id === semanticId);
    if (!anchor || !SEMANTIC_SCENE_ANCHORS.includes(semanticId)) return;
    setHistory((current) => [...current, anchors]);
    await perform("Removing crease anchor...", () =>
      saveSceneCalibrationAnchors(analysisId, result.anchor_version, [{
        ...anchorInput(anchor),
        video_point: null,
        source: "manually_added"
      }])
    );
  }

  async function undo() {
    if (!result || !history.length) return;
    const previous = history[history.length - 1];
    setHistory((current) => current.slice(0, -1));
    await perform("Restoring previous anchors...", () =>
      saveSceneCalibrationAnchors(
        analysisId,
        result.anchor_version,
        previous.map(anchorInput)
      )
    );
  }

  if (!result) {
    return <p className="text-sm text-white/45">Loading scene calibration...</p>;
  }

  if (result.stage === "NOT_STARTED") {
    return (
      <section aria-labelledby="scene-calibration-heading">
        <h2 id="scene-calibration-heading" className="text-xl font-black">Scene Calibration</h2>
        <p className="mt-2 text-sm text-white/55">
          Detect wickets to begin scene calibration.
        </p>
        {error && <p className="mt-3 text-sm text-[#ffaaa6]">{error}</p>}
        <Button className="mt-4" disabled={busy} onClick={() => void detect()}>
          {busy ? "Detecting Wickets..." : "Detect Wickets"}
        </Button>
      </section>
    );
  }

  return (
    <section aria-labelledby="scene-calibration-heading">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="scene-calibration-heading" className="text-xl font-black">Scene Calibration</h2>
          <p className="mt-1 text-sm text-white/55">{result.message}</p>
        </div>
        <div className="border border-white/10 bg-black/20 px-3 py-2 text-right">
          <p className="text-[10px] font-bold uppercase text-white/35">Quality</p>
          <p className="mt-1 text-sm font-black uppercase">{qualityLabel(result)}</p>
        </div>
      </div>

      {busy && (
        <div className="mt-4 border border-lime/20 bg-lime/[0.05] px-3 py-3">
          <p className="text-sm font-bold text-lime">{busyMessage}</p>
          <p className="mt-1 text-xs text-white/45">
            Detecting wickets / stabilising observations / extracting anchors / aligning pitch / validating
          </p>
        </div>
      )}
      {error && (
        <p className="mt-4 border border-signal/30 bg-signal/10 px-3 py-2 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}

      {setup && result.setup_frame_image_url && (
        <>
          <div className="mt-4">
            <AnalysisMediaStage
              aspectWidth={setup.image_width}
              aspectHeight={setup.image_height}
              label="Assisted scene calibration workspace"
              expandable
              stageRef={stageRef}
            >
              <div
                className="absolute inset-0 origin-center"
                style={{
                  transform: `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="absolute inset-0 h-full w-full select-none object-contain"
                  src={result.setup_frame_image_url}
                  alt={`Calibration setup frame ${setup.frame_index}`}
                  draggable={false}
                />
                <svg
                  className="absolute inset-0 h-full w-full"
                  viewBox={`0 0 ${setup.image_width} ${setup.image_height}`}
                  preserveAspectRatio="xMidYMid meet"
                  onPointerMove={moveDrag}
                  onPointerUp={(event) => void endDrag(event)}
                  onPointerCancel={(event) => void endDrag(event)}
                  onPointerDown={(event) => {
                    if (placing) void placeAnchor(event);
                  }}
                >
                  {showOverlay && projection && (
                    <PitchProjection
                      projection={projection}
                      opacity={opacity}
                      showProjectedAnchors={showProjectedAnchors}
                    />
                  )}
                  {showAutomatic && anchors.map((anchor) => (
                    anchor.original_automatic_point ? (
                      <circle
                        key={`auto-${anchor.semantic_id}`}
                        cx={anchor.original_automatic_point.x}
                        cy={anchor.original_automatic_point.y}
                        r="6"
                        fill="none"
                        stroke="#8f9a95"
                        strokeDasharray="3 2"
                        strokeWidth="2"
                        pointerEvents="none"
                      />
                    ) : null
                  ))}
                  {showAnchors && anchors.map((anchor) => (
                    anchor.video_point ? (
                      <g key={anchor.semantic_id}>
                        <circle
                          cx={anchor.video_point.x}
                          cy={anchor.video_point.y}
                          r={anchor.kind === "crease" ? 7 : 9}
                          fill={anchor.valid ? (anchor.kind === "crease" ? "#55d4f4" : "#b8f276") : "#ff6961"}
                          stroke="#07110d"
                          strokeWidth="3"
                          style={{
                            cursor: editing ? "grab" : "default",
                            touchAction: "none"
                          }}
                          onPointerDown={(event) => startDrag(event, anchor.semantic_id)}
                        />
                        <text
                          x={anchor.video_point.x + 12}
                          y={anchor.video_point.y - 10}
                          fill="#ffffff"
                          fontSize="15"
                          fontWeight="700"
                          pointerEvents="none"
                        >
                          {readable(anchor.semantic_id)}
                        </text>
                      </g>
                    ) : null
                  ))}
                </svg>
              </div>
            </AnalysisMediaStage>
          </div>

          <div className="mt-3 grid gap-3 border-y border-white/10 py-3 md:grid-cols-3">
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={showOverlay} onChange={(event) => setShowOverlay(event.target.checked)} />
              Virtual pitch
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={showAnchors} onChange={(event) => setShowAnchors(event.target.checked)} />
              Current anchors
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input type="checkbox" checked={showAutomatic} onChange={(event) => setShowAutomatic(event.target.checked)} />
              Automatic anchors
            </label>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={showProjectedAnchors}
                onChange={(event) => setShowProjectedAnchors(event.target.checked)}
              />
              Projected anchors
            </label>
            <label className="flex items-center gap-2 text-xs md:col-span-2">
              <span className="w-16">Opacity</span>
              <input
                className="min-w-0 flex-1"
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                aria-label="Calibration overlay opacity"
                onChange={(event) => setOpacity(Number(event.target.value))}
              />
            </label>
            <div className="flex flex-wrap items-center gap-1">
              <button title="Zoom out" aria-label="Zoom out" className="h-9 w-9 border border-white/15" onClick={() => setView((current) => ({ ...current, zoom: Math.max(1, current.zoom - 0.25) }))}>-</button>
              <span className="w-12 text-center text-xs">{view.zoom.toFixed(2)}x</span>
              <button title="Zoom in" aria-label="Zoom in" className="h-9 w-9 border border-white/15" onClick={() => setView((current) => ({ ...current, zoom: Math.min(3, current.zoom + 0.25) }))}>+</button>
              <button title="Pan left" aria-label="Pan left" disabled={view.zoom === 1} className="h-9 w-9 border border-white/15" onClick={() => setView((current) => ({ ...current, panX: current.panX - 20 }))}>{"<"}</button>
              <button title="Pan right" aria-label="Pan right" disabled={view.zoom === 1} className="h-9 w-9 border border-white/15" onClick={() => setView((current) => ({ ...current, panX: current.panX + 20 }))}>{">"}</button>
              <button title="Pan up" aria-label="Pan up" disabled={view.zoom === 1} className="h-9 w-9 border border-white/15" onClick={() => setView((current) => ({ ...current, panY: current.panY - 20 }))}>{"^"}</button>
              <button title="Pan down" aria-label="Pan down" disabled={view.zoom === 1} className="h-9 w-9 border border-white/15" onClick={() => setView((current) => ({ ...current, panY: current.panY + 20 }))}>{"v"}</button>
              <button title="Reset view" className="h-9 border border-white/15 px-2 text-xs" onClick={() => setView({ zoom: 1, panX: 0, panY: 0 })}>Reset view</button>
            </div>
          </div>
        </>
      )}

      {(result.competing_candidate || result.orientation_resolution) && (
        <div className="mt-4 border border-white/10 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-black">Mirror Candidate Comparison</h3>
              <p className="mt-1 text-xs text-white/45">
                Both overlays can align because wicket geometry is laterally symmetric.
              </p>
            </div>
            {result.competing_candidate && (
              <div className="flex gap-1">
                <button
                  className={`border px-2 py-1 text-xs ${candidateView === "selected" ? "border-lime text-lime" : "border-white/15"}`}
                  onClick={() => setCandidateView("selected")}
                >
                  Candidate A
                </button>
                <button
                  className={`border px-2 py-1 text-xs ${candidateView === "competing" ? "border-lime text-lime" : "border-white/15"}`}
                  onClick={() => setCandidateView("competing")}
                >
                  Candidate B
                </button>
              </div>
            )}
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            <CandidateSummary label="Candidate A" candidate={result.selected_candidate} active={candidateView === "selected"} />
            <CandidateSummary label="Candidate B" candidate={result.competing_candidate} active={candidateView === "competing"} />
          </div>
          <div className="mt-3 grid gap-2 text-xs text-white/55 sm:grid-cols-3">
            <span>Bowler end</span>
            <span className="text-center">Bowler to striker</span>
            <span className="text-right">Striker end</span>
            <span>Pitch Left</span>
            <span className="text-center">Native video orientation</span>
            <span className="text-right">Pitch Right</span>
          </div>
          {result.orientation_resolution && (
            <p className="mt-3 text-xs text-white/45">
              Ambiguity {result.orientation_resolution.ambiguity_before.toFixed(3)}
              {" -> "}
              {result.orientation_resolution.ambiguity_after.toFixed(3)}
              {" / "}
              {mappingText(result.orientation_resolution.image_left_mapping)}
            </p>
          )}
        </div>
      )}

      {orientationRequired && (
        <div className="mt-4 border border-[#ffd35f]/30 bg-[#ffd35f]/10 p-3">
          <h3 className="text-sm font-black text-[#ffd35f]">Pitch orientation needs confirmation</h3>
          <p className="mt-1 text-xs text-white/65">
            Choose which side of the native video represents pitch-right when looking from the bowler end toward the striker end.
          </p>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            <Button disabled={busy} onClick={() => void confirmOrientation("IMAGE_LEFT_IS_PITCH_LEFT")}>
              Image Left = Pitch Left
            </Button>
            <Button disabled={busy} onClick={() => void confirmOrientation("IMAGE_LEFT_IS_PITCH_RIGHT")}>
              Image Left = Pitch Right
            </Button>
            <Button variant="secondary" disabled={busy} onClick={() => void confirmOrientation("NOT_SURE")}>
              Not Sure
            </Button>
          </div>
          <label className="mt-3 flex items-center gap-2 text-xs text-white/55">
            <input
              type="checkbox"
              checked={savePreset}
              onChange={(event) => setSavePreset(event.target.checked)}
            />
            Save this fixed-camera orientation for explicit future reuse
          </label>
        </div>
      )}

      {result.image_left_mapping && !orientationRequired && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border border-lime/20 bg-lime/[0.05] px-3 py-2 text-xs">
          <span>Orientation resolved: {mappingText(result.image_left_mapping)}</span>
          <button className="border border-white/15 px-2 py-1" onClick={() => void clearOrientation()}>
            Clear
          </button>
        </div>
      )}

      {editing && (
        <div className="mt-4 border border-white/10 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-black">Wicket Anchors</h3>
            <div className="flex flex-wrap gap-2">
              <Button variant="secondary" disabled={!history.length || busy} onClick={() => void undo()}>Undo</Button>
              <Button variant="secondary" disabled={busy} onClick={() => void resetAll()}>Reset All</Button>
            </div>
          </div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {REQUIRED_ANCHORS.map((semanticId) => {
              const anchor = anchors.find((item) => item.semantic_id === semanticId);
              return (
                <div key={semanticId} className="flex min-w-0 items-center justify-between gap-2 border-t border-white/10 py-2 text-xs">
                  <div className="min-w-0">
                    <p className="truncate font-bold">{readable(semanticId)}</p>
                    <p className={anchor?.valid ? "text-white/35" : "text-[#ffaaa6]"}>
                      {anchor?.video_point
                        ? `${anchor.video_point.x.toFixed(1)}, ${anchor.video_point.y.toFixed(1)} / ${readable(anchor.source)}`
                        : "Unavailable"}
                    </p>
                  </div>
                  {anchor?.original_automatic_point ? (
                    <button className="border border-white/15 px-2 py-1" onClick={() => void resetAnchor(semanticId)}>Reset</button>
                  ) : (
                    <button className="border border-white/15 px-2 py-1" onClick={() => setPlacing(semanticId)}>Add</button>
                  )}
                </div>
              );
            })}
          </div>
          <div className="mt-4 border-t border-white/10 pt-3">
            <h3 className="text-sm font-black">Optional Semantic Scene Anchors</h3>
            <div className="mt-2 flex flex-wrap gap-2">
              {SEMANTIC_SCENE_ANCHORS.map((semanticId) => {
                const exists = anchors.some((item) => item.semantic_id === semanticId && item.video_point);
                return (
                  <button
                    key={semanticId}
                    className="border border-white/15 px-2 py-1.5 text-xs"
                    onClick={() => exists
                      ? void removeOptionalAnchor(semanticId)
                      : setPlacing(semanticId)}
                  >
                    {exists ? `Remove ${readable(semanticId)}` : `Add ${readable(semanticId)}`}
                  </button>
                );
              })}
            </div>
          </div>
          {placing && (
            <p className="mt-3 bg-[#55d4f4]/10 px-3 py-2 text-xs text-[#a9eafb]">
              Click the video to place {readable(placing)}.
            </p>
          )}
          {requiredUnavailable.length > 0 && (
            <p className="mt-3 text-xs text-[#ffaaa6]">
              Missing: {requiredUnavailable.map(readable).join(", ")}
            </p>
          )}
        </div>
      )}

      {result.validation && (
        <details className="mt-4 border border-white/10 p-3">
          <summary className="cursor-pointer text-sm font-bold">
            Backend Validation ({result.validation.checks.filter((item) => item.passed).length}/{result.validation.checks.length})
          </summary>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {result.validation.checks.map((check) => (
              <div key={check.threshold_id} className="border-t border-white/10 py-2 text-xs">
                <strong className={check.passed ? "text-lime" : "text-[#ffaaa6]"}>
                  {check.passed ? "Pass" : "Fail"} / {readable(check.threshold_id)}
                </strong>
                <p className="mt-1 text-white/45">{check.reason}</p>
              </div>
            ))}
          </div>
        </details>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        <Button variant="secondary" disabled={busy} onClick={() => void detect()}>Redetect</Button>
        <Button
          variant="secondary"
          disabled={!canEdit || busy}
          onClick={() => setEditing((value) => !value)}
        >
          {editing ? "Close Fine Adjust" : "Fine Adjust"}
        </Button>
        <Button
          disabled={!canEdit || busy || requiredUnavailable.length > 0}
          onClick={() => void perform("Recalculating alignment...", () =>
            refineSceneCalibration(analysisId, result.anchor_version)
          )}
        >
          Recalculate Alignment
        </Button>
        <Button
          disabled={!candidateReady || busy}
          onClick={() => void perform("Accepting validated calibration...", () =>
            acceptSceneCalibration(
              analysisId,
              result.anchor_version,
              result.selected_candidate?.candidate_id
            )
          )}
        >
          Accept Calibration
        </Button>
        <Button
          variant="danger"
          disabled={!result.selected_candidate || busy}
          onClick={() => void perform("Rejecting calibration...", () =>
            rejectSceneCalibration(analysisId, result.anchor_version)
          )}
        >
          Reject Calibration
        </Button>
        {result.calibration_level === "VISUAL_ONLY" && result.selected_candidate && (
          <Button
            variant="secondary"
            disabled={busy}
            onClick={() => void perform("Enabling visual overlay only...", () =>
              enableVisualSceneCalibration(analysisId, result.anchor_version)
            )}
          >
            Use Visual Overlay Only
          </Button>
        )}
        {terminal && (
          <Button variant="secondary" onClick={onContinue}>Continue</Button>
        )}
      </div>

      <div className="mt-4 flex flex-wrap gap-x-4 gap-y-2 border-t border-white/10 pt-3 text-xs text-white/45">
        <span>Stage: {readable(result.stage)}</span>
        <span>Anchor version: {result.anchor_version}</span>
        <span>{result.metrics_unlocked.length} metric groups unlocked</span>
        {result.accepted_calibration && (
          <a className="text-lime underline" href={result.accepted_calibration.snapshot_url} target="_blank" rel="noreferrer">
            Accepted revision {result.accepted_calibration.revision}
          </a>
        )}
      </div>
    </section>
  );
}
