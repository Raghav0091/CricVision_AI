"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent
} from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  acceptWicketBoxCalibration,
  detectWicketBoxCalibration,
  registerWicketBoxCalibration,
  type VideoAnalysisPreparedResponse
} from "@/lib/api";
import {
  displayToVideoNative,
  videoNativeToDisplay,
  type CalibrationViewTransform
} from "@/lib/calibrationCoordinates";
import { getDeviceId } from "@/lib/deviceIdentity";
import {
  clampNativePoint,
  defaultWicketBoxes,
  normalizedToWicketBox,
  wicketBoxToNormalized,
  type NormalizedBox
} from "@/lib/wicketCalibration/coordinates";
import {
  boxCursor,
  canStartBoxDrag,
  interactionModeAfterPointerUp,
  matchesPointerSession,
  resolveInteractionMode,
  shouldContinuePointerMove,
  type InteractionMode,
  type PointerSession,
  type ResizeHandle
} from "@/lib/wicketCalibration/pointerInteraction";
import type {
  CalibrationResult,
  CricketPitchGeometry,
  StumpIdentity,
  StumpLandmark,
  WicketBox,
  WicketBoxCalibrationCandidateSummary,
  WicketBoxRole
} from "@/lib/wicketCalibration/types";

import { AnalysisMediaStage } from "./AnalysisMediaStage";


type Phase =
  | "boxes"
  | "detecting"
  | "registering"
  | "preview"
  | "review"
  | "accepted"
  | "failed";

type ResizeCorner = ResizeHandle;

type PointerOperation = PointerSession & {
  startX: number;
  startY: number;
  original: NormalizedBox;
};

type LandmarkEndpoint = "base" | "top";

type LandmarkPointerOperation = {
  pointerId: number;
  role: WicketBoxRole;
  identity: StumpIdentity;
  endpoint: LandmarkEndpoint;
  startX: number;
  startY: number;
  original: { x: number; y: number };
};

const MIN_BOX_SIZE = 0.035;
const VIEW_TRANSFORM: CalibrationViewTransform = { zoom: 1, panX: 0, panY: 0 };


function formatPx(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(2)} px`;
}


function formatMeters(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(2)} m`;
}


function CandidateTechnicalDetails({
  candidate,
  label
}: {
  candidate: WicketBoxCalibrationCandidateSummary;
  label: string;
}) {
  return (
    <details className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <summary className="cursor-pointer text-sm font-semibold text-white/75">
        {label} · Technical details
      </summary>
      <dl className="mt-3 grid gap-2 text-xs text-white/55 sm:grid-cols-2">
        <div><dt className="text-white/35">Hypothesis</dt><dd>{candidate.assignment_hypothesis}</dd></div>
        <div><dt className="text-white/35">Reprojection RMSE</dt><dd>{formatPx(candidate.reprojection_rmse_px)}</dd></div>
        <div><dt className="text-white/35">NEAR wicket error</dt><dd>{formatPx(candidate.near_wicket_error_px)}</dd></div>
        <div><dt className="text-white/35">FAR wicket error</dt><dd>{formatPx(candidate.far_wicket_error_px)}</dd></div>
        <div><dt className="text-white/35">Camera height</dt><dd>{formatMeters(candidate.camera_height_m)}</dd></div>
        <div><dt className="text-white/35">Focal length</dt><dd>{formatPx(candidate.focal_length_px)}</dd></div>
        <div><dt className="text-white/35">Stability</dt><dd>{candidate.stability_score?.toFixed(2) ?? "—"}</dd></div>
        <div><dt className="text-white/35">Semantic ends</dt><dd>NEAR={candidate.near_semantic_end}, FAR={candidate.far_semantic_end}</dd></div>
      </dl>
      {candidate.rejection_reasons.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs text-[#ffb4b0]">
          {candidate.rejection_reasons.map((reason) => (
            <li key={reason}>• {reason}</li>
          ))}
        </ul>
      )}
    </details>
  );
}


function phaseBadge(phase: Phase): { label: string; tone: "neutral" | "good" | "warn" } {
  if (phase === "accepted") return { label: "Accepted", tone: "good" };
  if (phase === "preview") return { label: "Ready", tone: "good" };
  if (phase === "detecting" || phase === "registering") {
    return { label: "Processing", tone: "neutral" };
  }
  if (phase === "failed" || phase === "review") {
    return { label: "Needs Attention", tone: "warn" };
  }
  return { label: "Step 1", tone: "neutral" };
}


function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}


function moveBox(box: NormalizedBox, dx: number, dy: number): NormalizedBox {
  return {
    ...box,
    x: clamp(box.x + dx, 0, 1 - box.width),
    y: clamp(box.y + dy, 0, 1 - box.height)
  };
}


function resizeBox(
  box: NormalizedBox,
  corner: ResizeCorner,
  dx: number,
  dy: number
): NormalizedBox {
  let { x, y, width, height } = box;
  if (corner.includes("e")) width = clamp(width + dx, MIN_BOX_SIZE, 1 - x);
  if (corner.includes("w")) {
    const nextX = clamp(x + dx, 0, x + width - MIN_BOX_SIZE);
    width = clamp(width - (nextX - x), MIN_BOX_SIZE, 1);
    x = nextX;
  }
  if (corner.includes("s")) height = clamp(height + dy, MIN_BOX_SIZE, 1 - y);
  if (corner.includes("n")) {
    const nextY = clamp(y + dy, 0, y + height - MIN_BOX_SIZE);
    height = clamp(height - (nextY - y), MIN_BOX_SIZE, 1);
    y = nextY;
  }
  return { x, y, width, height };
}


export function WicketBoxCalibrationPanel({
  analysis,
  onAccepted,
  onContinue
}: {
  analysis: VideoAnalysisPreparedResponse;
  onAccepted?: (calibration: CalibrationResult) => void;
  onContinue?: () => void;
}) {
  // Uploaded footage is always a real pitch, so this stays null — the solver
  // reads null as regulation 22 yards. Deliberately not restored from the
  // /live rig settings: a 1.5m indoor rig leaking into a net video would
  // silently scale every speed by more than 13x.
  const pitchGeometry: CricketPitchGeometry | null = null;
  const stageRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const operationRef = useRef<PointerOperation | null>(null);
  const landmarkOperationRef = useRef<LandmarkPointerOperation | null>(null);
  const [phase, setPhase] = useState<Phase>("boxes");
  const [frameIndex, setFrameIndex] = useState(analysis.reference_frame_index);
  const [frameUrl, setFrameUrl] = useState(analysis.reference_frame_url);
  const [nearGuide, setNearGuide] = useState<NormalizedBox>(
    wicketBoxToNormalized(
      defaultWicketBoxes(analysis.width, analysis.height, frameIndex).NEAR
    )
  );
  const [farGuide, setFarGuide] = useState<NormalizedBox>(
    wicketBoxToNormalized(
      defaultWicketBoxes(analysis.width, analysis.height, frameIndex).FAR
    )
  );
  const [landmarks, setLandmarks] = useState<StumpLandmark[]>([]);
  const [automaticLandmarks, setAutomaticLandmarks] = useState<StumpLandmark[]>([]);
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [message, setMessage] = useState(
    "Draw NEAR and FAR wicket boxes on the calibration frame, then Detect."
  );
  const [error, setError] = useState<string | null>(null);
  const [orientationRequired, setOrientationRequired] = useState(false);
  const [selectedHypothesis, setSelectedHypothesis] = useState<"A" | "B" | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("SELECT");
  const [activeBox, setActiveBox] = useState<WicketBoxRole | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 640, height: 360 });
  const busy = phase === "detecting" || phase === "registering";
  const boxesEditable = phase === "boxes" && !busy;
  const resolvedInteractionMode = resolveInteractionMode(boxesEditable, interactionMode);
  const registrationReady = Boolean(
    calibration?.registration_summary?.recommended?.physically_valid
  );
  const needsManualCorrection = (
    phase === "review"
    || phase === "failed"
    || Boolean(error)
    || Boolean(calibration?.registration_summary?.rejected.length)
  ) && !registrationReady;
  const landmarksEditable = needsManualCorrection && !busy && landmarks.length > 0;
  const automaticLandmarksValid = (
    landmarks.length >= 12
    && landmarks.every((item) => item.provenance === "AUTOMATIC")
    && registrationReady
  );
  const badge = phaseBadge(phase);

  const nearBox = useMemo(
    () => normalizedToWicketBox(
      nearGuide,
      "NEAR",
      analysis.width,
      analysis.height,
      frameIndex
    ),
    [analysis.height, analysis.width, frameIndex, nearGuide]
  );
  const farBox = useMemo(
    () => normalizedToWicketBox(
      farGuide,
      "FAR",
      analysis.width,
      analysis.height,
      frameIndex
    ),
    [analysis.height, analysis.width, frameIndex, farGuide]
  );

  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setContainerSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const clearPointerSession = useCallback((pointerId?: number) => {
    const operation = operationRef.current;
    const landmarkOperation = landmarkOperationRef.current;
    const captureTarget = overlayRef.current;
    if (operation && (pointerId === undefined || operation.pointerId === pointerId)) {
      if (captureTarget?.hasPointerCapture(operation.pointerId)) {
        captureTarget.releasePointerCapture(operation.pointerId);
      }
      operationRef.current = null;
      setActiveBox(null);
      setInteractionMode((current) => interactionModeAfterPointerUp(
        resolveInteractionMode(boxesEditable, current)
      ));
    }
    if (
      landmarkOperation
      && (pointerId === undefined || landmarkOperation.pointerId === pointerId)
    ) {
      if (captureTarget?.hasPointerCapture(landmarkOperation.pointerId)) {
        captureTarget.releasePointerCapture(landmarkOperation.pointerId);
      }
      landmarkOperationRef.current = null;
    }
  }, [boxesEditable]);

  useEffect(() => {
    function handleWindowPointerEnd(event: PointerEvent) {
      clearPointerSession(event.pointerId);
    }
    function handleWindowBlur() {
      clearPointerSession();
    }
    window.addEventListener("pointerup", handleWindowPointerEnd);
    window.addEventListener("pointercancel", handleWindowPointerEnd);
    window.addEventListener("blur", handleWindowBlur);
    return () => {
      window.removeEventListener("pointerup", handleWindowPointerEnd);
      window.removeEventListener("pointercancel", handleWindowPointerEnd);
      window.removeEventListener("blur", handleWindowBlur);
    };
  }, [clearPointerSession]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    function handleLostCapture(event: PointerEvent) {
      if (
        matchesPointerSession(operationRef.current, event.pointerId)
        || landmarkOperationRef.current?.pointerId === event.pointerId
      ) {
        clearPointerSession(event.pointerId);
      }
    }
    overlay.addEventListener("lostpointercapture", handleLostCapture);
    return () => overlay.removeEventListener("lostpointercapture", handleLostCapture);
  }, [clearPointerSession]);

  function resetBoxes() {
    const defaults = defaultWicketBoxes(analysis.width, analysis.height, frameIndex);
    setNearGuide(wicketBoxToNormalized(defaults.NEAR));
    setFarGuide(wicketBoxToNormalized(defaults.FAR));
    setLandmarks([]);
    setAutomaticLandmarks([]);
    setCalibration(null);
    setOrientationRequired(false);
    setSelectedHypothesis(null);
    setInteractionMode("SELECT");
    setActiveBox(null);
    operationRef.current = null;
    landmarkOperationRef.current = null;
    setPhase("boxes");
    setError(null);
    setMessage("Reset to default NEAR/FAR boxes.");
  }

  function buildRequest(extraLandmarks = landmarks) {
    return {
      analysis_id: analysis.analysis_id,
      calibration_frame_index: frameIndex,
      source_image_width: analysis.width,
      source_image_height: analysis.height,
      near_wicket_box: nearBox,
      far_wicket_box: farBox,
      stump_landmarks: extraLandmarks,
      // Only honoured when a profile exists for this browser at a matching
      // aspect ratio; footage from another camera falls back to the sweep. The
      // lens badge shows which of the two actually happened.
      device_id: getDeviceId(),
      // Null means regulation, which is what full-size footage sends.
      pitch_geometry: pitchGeometry
    };
  }

  async function runRegister(
    hypothesis: "A" | "B" | null = selectedHypothesis,
    landmarkOverride?: StumpLandmark[]
  ) {
    setPhase("registering");
    setError(null);
    try {
      const response = await registerWicketBoxCalibration(
        analysis.analysis_id,
        buildRequest(landmarkOverride ?? landmarks),
        hypothesis
      );
      if (!response.success) {
        setCalibration(response.calibration ?? null);
        setOrientationRequired(
          Boolean(
            response.calibration?.warnings?.includes("orientation_ambiguous")
            || response.calibration?.registration_summary?.orientation_ambiguous
          )
        );
        if (response.calibration?.registration_summary?.recommended) {
          setSelectedHypothesis(
            response.calibration.registration_summary.recommended.assignment_hypothesis
          );
        }
        setPhase("review");
        setError(response.message);
        setMessage(response.message);
        return;
      }
      setCalibration(response.calibration ?? null);
      setOrientationRequired(
        Boolean(response.calibration?.registration_summary?.orientation_ambiguous)
      );
      if (response.calibration?.registration_summary?.recommended) {
        setSelectedHypothesis(
          response.calibration.registration_summary.recommended.assignment_hypothesis
        );
      } else {
        setSelectedHypothesis(null);
      }
      setPhase("preview");
      setMessage(
        landmarks.every((item) => item.provenance === "AUTOMATIC")
          ? "Stumps detected automatically. No manual correction required."
          : (response.message || "Registration ready for acceptance.")
      );
    } catch (caught) {
      setPhase("failed");
      setError(caught instanceof Error ? caught.message : "Register failed.");
    }
  }

  async function runDetect() {
    setPhase("detecting");
    setError(null);
    try {
      const response = await detectWicketBoxCalibration(
        analysis.analysis_id,
        buildRequest([])
      );
      if (!response.success) {
        setPhase("failed");
        setError(response.message);
        setMessage(response.message);
        return;
      }
      setLandmarks(response.stump_landmarks);
      setAutomaticLandmarks(response.stump_landmarks);
      await runRegister(null, response.stump_landmarks);
    } catch (caught) {
      setPhase("failed");
      setError(caught instanceof Error ? caught.message : "Detect failed.");
    }
  }

  async function runAccept(hypothesis: "A" | "B" | null = selectedHypothesis) {
    setPhase("registering");
    setError(null);
    try {
      const response = await acceptWicketBoxCalibration(
        analysis.analysis_id,
        {
          analysis_id: analysis.analysis_id,
          accept_registered_calibration: true
        },
        hypothesis
      );
      if (!response.success) {
        setCalibration(response.calibration ?? null);
        setPhase("review");
        setError(response.message);
        setMessage(response.message);
        return;
      }
      setCalibration(response.calibration ?? null);
      setPhase("accepted");
      setMessage(response.message || "Wicket-box calibration accepted.");
      if (response.calibration) onAccepted?.(response.calibration);
    } catch (caught) {
      setPhase("failed");
      setError(caught instanceof Error ? caught.message : "Accept failed.");
    }
  }

  function updateLandmarkPoint(
    role: WicketBoxRole,
    identity: StumpIdentity,
    endpoint: LandmarkEndpoint,
    nativePoint: { x: number; y: number }
  ) {
    const clamped = clampNativePoint(nativePoint, analysis.width, analysis.height);
    setLandmarks((previous) => previous.map((item) => {
      if (item.wicket_role !== role || item.stump_identity !== identity) return item;
      const nextPoint = { x: clamped.x, y: clamped.y };
      const base = endpoint === "base" ? nextPoint : item.base;
      const top = endpoint === "top" ? nextPoint : item.top;
      return {
        ...item,
        base,
        top,
        centre: {
          x: (base.x + top.x) / 2,
          y: (base.y + top.y) / 2
        },
        provenance: "USER_CORRECTED"
      };
    }));
  }

  function startLandmarkOperation(
    event: ReactPointerEvent<SVGElement>,
    role: WicketBoxRole,
    identity: StumpIdentity,
    endpoint: LandmarkEndpoint,
    point: { x: number; y: number }
  ) {
    if (!landmarksEditable) return;
    event.preventDefault();
    event.stopPropagation();
    const stage = overlayRef.current;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    landmarkOperationRef.current = {
      pointerId: event.pointerId,
      role,
      identity,
      endpoint,
      startX: event.clientX - rect.left,
      startY: event.clientY - rect.top,
      original: { ...point }
    };
    stage.setPointerCapture(event.pointerId);
  }

  function updateGuide(role: WicketBoxRole, box: NormalizedBox) {
    if (role === "NEAR") setNearGuide(box);
    else setFarGuide(box);
  }

  function startOperation(
    event: ReactPointerEvent<HTMLElement>,
    role: WicketBoxRole,
    box: NormalizedBox,
    mode: PointerOperation["mode"],
    corner?: ResizeCorner
  ) {
    if (!canStartBoxDrag(resolvedInteractionMode, role)) return;
    event.preventDefault();
    event.stopPropagation();
    const overlay = overlayRef.current;
    if (!overlay) return;
    const rect = overlay.getBoundingClientRect();
    operationRef.current = {
      pointerId: event.pointerId,
      role,
      mode,
      handle: corner,
      startX: event.clientX - rect.left,
      startY: event.clientY - rect.top,
      original: { ...box }
    };
    setActiveBox(role);
    overlay.setPointerCapture(event.pointerId);
  }

  function moveOperation(event: ReactPointerEvent<HTMLDivElement>) {
    if (!shouldContinuePointerMove(event.buttons)) {
      clearPointerSession(event.pointerId);
      return;
    }

    const landmarkOperation = landmarkOperationRef.current;
    const overlay = overlayRef.current;
    if (landmarkOperation && overlay && event.pointerId === landmarkOperation.pointerId && landmarksEditable) {
      event.preventDefault();
      const rect = overlay.getBoundingClientRect();
      const currentNative = displayToVideoNative(
        { x: event.clientX - rect.left, y: event.clientY - rect.top },
        { width: analysis.width, height: analysis.height },
        containerSize,
        VIEW_TRANSFORM
      );
      if (currentNative) {
        updateLandmarkPoint(
          landmarkOperation.role,
          landmarkOperation.identity,
          landmarkOperation.endpoint,
          currentNative
        );
      }
      return;
    }

    const operation = operationRef.current;
    if (
      !operation
      || !overlay
      || !matchesPointerSession(operation, event.pointerId)
      || !canStartBoxDrag(resolvedInteractionMode, operation.role)
    ) {
      return;
    }
    event.preventDefault();
    const rect = overlay.getBoundingClientRect();
    const currentX = event.clientX - rect.left;
    const currentY = event.clientY - rect.top;
    const startNative = displayToVideoNative(
      { x: operation.startX, y: operation.startY },
      { width: analysis.width, height: analysis.height },
      containerSize,
      VIEW_TRANSFORM
    );
    const currentNative = displayToVideoNative(
      { x: currentX, y: currentY },
      { width: analysis.width, height: analysis.height },
      containerSize,
      VIEW_TRANSFORM
    );
    if (!startNative || !currentNative) return;
    const dx = (currentNative.x - startNative.x) / analysis.width;
    const dy = (currentNative.y - startNative.y) / analysis.height;
    const next = operation.mode === "move"
      ? moveBox(operation.original, dx, dy)
      : resizeBox(operation.original, operation.handle ?? "se", dx, dy);
    updateGuide(operation.role, next);
  }

  function endOperation(event: ReactPointerEvent<HTMLDivElement>) {
    clearPointerSession(event.pointerId);
  }

  function renderBox(role: WicketBoxRole, box: NormalizedBox, colour: string) {
    const editable = canStartBoxDrag(resolvedInteractionMode, role);
    const topLeft = videoNativeToDisplay(
      { x: box.x * analysis.width, y: box.y * analysis.height },
      { width: analysis.width, height: analysis.height },
      containerSize,
      VIEW_TRANSFORM
    );
    const bottomRight = videoNativeToDisplay(
      {
        x: (box.x + box.width) * analysis.width,
        y: (box.y + box.height) * analysis.height
      },
      { width: analysis.width, height: analysis.height },
      containerSize,
      VIEW_TRANSFORM
    );
    const left = topLeft.x;
    const top = topLeft.y;
    const width = Math.max(8, bottomRight.x - topLeft.x);
    const height = Math.max(8, bottomRight.y - topLeft.y);
    return (
      <div
        key={role}
        className="absolute rounded-md border-2 border-dashed"
        style={{
          left,
          top,
          width,
          height,
          borderColor: colour,
          boxShadow: `0 0 16px ${colour}55`,
          cursor: boxCursor(resolvedInteractionMode, role),
          touchAction: editable ? "none" : "auto",
          pointerEvents: editable ? "auto" : "none",
          opacity: activeBox && activeBox !== role ? 0.72 : 1
        }}
        onPointerDown={(event) => startOperation(event, role, box, "move")}
      >
        <span
          className="absolute -top-7 left-0 rounded px-2 py-1 text-[10px] font-black uppercase tracking-wide text-white"
          style={{ backgroundColor: colour, pointerEvents: "none" }}
        >
          {role}
        </span>
        {editable && (["nw", "ne", "sw", "se"] as ResizeCorner[]).map((corner) => (
          <button
            key={corner}
            type="button"
            aria-label={`Resize ${role} box`}
            className={`absolute h-4 w-4 rounded-sm border-2 border-ink ${
              corner.includes("n") ? "-top-2" : "-bottom-2"
            } ${corner.includes("w") ? "-left-2" : "-right-2"}`}
            style={{
              backgroundColor: colour,
              cursor: boxCursor(resolvedInteractionMode, role, corner),
              touchAction: "none"
            }}
            onPointerDown={(event) => startOperation(event, role, box, "resize", corner)}
          />
        ))}
      </div>
    );
  }

  function renderLandmarks() {
    return landmarks.flatMap((item) => {
      const colour = item.wicket_role === "NEAR" ? "#ff554f" : "#ffd35f";
      const endpoints: Array<{ key: LandmarkEndpoint; point: { x: number; y: number }; label: string }> = [
        { key: "base", point: item.base, label: "B" },
        { key: "top", point: item.top, label: "T" }
      ];
      return endpoints.map(({ key, point, label }) => {
        const display = videoNativeToDisplay(
          point,
          { width: analysis.width, height: analysis.height },
          containerSize,
          VIEW_TRANSFORM
        );
        return (
          <g key={`${item.wicket_role}-${item.stump_identity}-${key}`}>
            <line
              x1={videoNativeToDisplay(
                item.base,
                { width: analysis.width, height: analysis.height },
                containerSize,
                VIEW_TRANSFORM
              ).x}
              y1={videoNativeToDisplay(
                item.base,
                { width: analysis.width, height: analysis.height },
                containerSize,
                VIEW_TRANSFORM
              ).y}
              x2={videoNativeToDisplay(
                item.top,
                { width: analysis.width, height: analysis.height },
                containerSize,
                VIEW_TRANSFORM
              ).x}
              y2={videoNativeToDisplay(
                item.top,
                { width: analysis.width, height: analysis.height },
                containerSize,
                VIEW_TRANSFORM
              ).y}
              stroke={colour}
              strokeWidth={1.5}
              strokeOpacity={0.7}
            />
            <circle
              cx={display.x}
              cy={display.y}
              r={landmarksEditable ? 7 : 4}
              fill={item.provenance === "USER_CORRECTED" ? "#6ed4a1" : colour}
              stroke="#08110d"
              strokeWidth={1.5}
              style={{
                cursor: landmarksEditable ? "grab" : "default",
                touchAction: landmarksEditable ? "none" : "auto",
                pointerEvents: landmarksEditable ? "auto" : "none"
              }}
              onPointerDown={(event) => startLandmarkOperation(
                event,
                item.wicket_role,
                item.stump_identity,
                key,
                point
              )}
            />
            {landmarksEditable && (
              <text
                x={display.x + 9}
                y={display.y + 4}
                fill="#ffffff"
                fontSize={9}
                fontWeight={700}
              >
                {item.stump_identity[0]}{label}
              </text>
            )}
          </g>
        );
      });
    });
  }

  return (
    <Card className="space-y-4 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-white/35">
            Scene Setup · Wicket Boxes
          </p>
          <h2 className="mt-1 text-lg font-black">Two-wicket box calibration</h2>
          <p className="mt-1 text-sm text-white/45">{message}</p>
        </div>
        <StatusBadge label={badge.label} tone={badge.tone} />
      </div>

      <div
        className="flex flex-wrap items-end gap-3"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <label className="text-xs text-white/55">
          Calibration frame
          <input
            className="mt-1 block rounded-lg border border-white/15 bg-black/30 px-3 py-2 text-sm"
            type="number"
            min={0}
            max={Math.max(analysis.frame_count - 1, 0)}
            value={frameIndex}
            disabled={busy || phase === "accepted"}
            onChange={(event) => {
              const next = Number(event.target.value);
              setFrameIndex(next);
              setFrameUrl(`${analysis.reference_frame_url}?frame=${next}`);
            }}
          />
        </label>
        {boxesEditable && (
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant={resolvedInteractionMode === "EDIT_NEAR" ? "primary" : "secondary"}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => {
                clearPointerSession();
                setInteractionMode("EDIT_NEAR");
                setActiveBox("NEAR");
              }}
            >
              Edit NEAR
            </Button>
            <Button
              type="button"
              variant={resolvedInteractionMode === "EDIT_FAR" ? "primary" : "secondary"}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => {
                clearPointerSession();
                setInteractionMode("EDIT_FAR");
                setActiveBox("FAR");
              }}
            >
              Edit FAR
            </Button>
            <Button
              type="button"
              variant="secondary"
              disabled={resolvedInteractionMode === "SELECT"}
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => {
                clearPointerSession();
                setInteractionMode("SELECT");
                setActiveBox(null);
              }}
            >
              Done editing
            </Button>
            <Button
              type="button"
              variant="secondary"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => {
                clearPointerSession();
                const defaults = defaultWicketBoxes(analysis.width, analysis.height, frameIndex);
                setNearGuide(wicketBoxToNormalized(defaults.NEAR));
                setInteractionMode("SELECT");
                setActiveBox(null);
              }}
            >
              Reset NEAR
            </Button>
            <Button
              type="button"
              variant="secondary"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={() => {
                clearPointerSession();
                const defaults = defaultWicketBoxes(analysis.width, analysis.height, frameIndex);
                setFarGuide(wicketBoxToNormalized(defaults.FAR));
                setInteractionMode("SELECT");
                setActiveBox(null);
              }}
            >
              Reset FAR
            </Button>
          </div>
        )}
      </div>

      <AnalysisMediaStage
        aspectWidth={analysis.width}
        aspectHeight={analysis.height}
        expandable
        label="Wicket box calibration frame"
        stageRef={stageRef}
      >
        <div
          ref={overlayRef}
          className="absolute inset-0 select-none"
          style={{
            touchAction: boxesEditable ? "none" : "auto",
            cursor: resolvedInteractionMode === "LOCKED" ? "default" : undefined
          }}
          onPointerMove={moveOperation}
          onPointerUp={endOperation}
          onPointerCancel={endOperation}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="absolute inset-0 h-full w-full object-contain"
            src={frameUrl}
            alt={`Calibration frame ${frameIndex}`}
            draggable={false}
          />
          <svg
            className={`absolute inset-0 h-full w-full ${landmarksEditable ? "" : "pointer-events-none"}`}
          >
            {renderLandmarks()}
            {calibration?.reprojection_diagnostics.map((item) => {
              const display = videoNativeToDisplay(
                { x: item.reprojected_pixel_x, y: item.reprojected_pixel_y },
                { width: analysis.width, height: analysis.height },
                containerSize,
                VIEW_TRANSFORM
              );
              return (
                <circle
                  key={`reproj-${item.landmark_id}`}
                  cx={display.x}
                  cy={display.y}
                  r={3}
                  fill="none"
                  stroke="#6ed4a1"
                  strokeWidth={2}
                />
              );
            })}
          </svg>
          {renderBox("FAR", farGuide, "#ffd35f")}
          {renderBox("NEAR", nearGuide, "#ff554f")}
        </div>
      </AnalysisMediaStage>

      {error && (
        <p className="rounded-lg border border-signal/35 bg-signal/10 px-3 py-2 text-sm text-[#ffb4b0]">
          {error}
        </p>
      )}

      {calibration?.registration_summary && (
        <div className="space-y-3 rounded-lg border border-white/10 bg-white/[0.03] p-3">
          <p className="text-sm text-white/75">
            {calibration.registration_summary.user_message}
          </p>
          {calibration.registration_summary.recommended && (
            <div className="rounded-lg border border-lime/25 bg-lime/[0.05] p-3">
              <p className="text-sm font-semibold text-lime">Recommended calibration</p>
              <p className="mt-1 text-xs text-white/55">
                Hypothesis {calibration.registration_summary.recommended.assignment_hypothesis}
                {" · "}
                RMSE {formatPx(calibration.registration_summary.recommended.reprojection_rmse_px)}
              </p>
              <CandidateTechnicalDetails
                candidate={calibration.registration_summary.recommended}
                label="Recommended calibration"
              />
            </div>
          )}
          {calibration.registration_summary.alternative && (
            <div className="rounded-lg border border-white/15 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold text-white/80">Alternative calibration</p>
                  <p className="mt-1 text-xs text-white/45">
                    Optional override when both solutions are physically valid.
                  </p>
                </div>
                <Button
                  type="button"
                  variant={
                    selectedHypothesis
                      === calibration.registration_summary.alternative.assignment_hypothesis
                      ? "primary"
                      : "secondary"
                  }
                  disabled={busy}
                  onPointerDown={(event) => event.stopPropagation()}
                  onClick={() => {
                    const hypothesis =
                      calibration.registration_summary?.alternative?.assignment_hypothesis;
                    if (!hypothesis) return;
                    setSelectedHypothesis(hypothesis);
                    void runRegister(hypothesis);
                  }}
                >
                  Use alternative
                </Button>
              </div>
              <CandidateTechnicalDetails
                candidate={calibration.registration_summary.alternative}
                label="Alternative calibration"
              />
            </div>
          )}
          {calibration.registration_summary.rejected.length > 0 && (
            <div className="rounded-lg border border-signal/25 bg-signal/5 p-3">
              <p className="text-sm font-semibold text-[#ffb4b0]">Rejected solution</p>
              <ul className="mt-2 space-y-2">
                {calibration.registration_summary.rejected.map((candidate) => (
                  <li key={candidate.candidate_id}>
                    <CandidateTechnicalDetails
                      candidate={candidate}
                      label={`Rejected · hypothesis ${candidate.assignment_hypothesis}`}
                    />
                  </li>
                ))}
              </ul>
            </div>
          )}
          {!calibration.registration_summary.recommended && (
            <p className="text-sm text-[#ffb4b0]">
              Calibration could not be accepted. Correct the stump points or adjust the wicket
              boxes and try again.
            </p>
          )}
        </div>
      )}

      {orientationRequired && calibration?.registration_summary?.alternative && (
        <p className="text-xs text-white/45">
          NEAR/FAR box labels stay fixed. Choose the alternative only if the recommended
          camera orientation does not match your scene.
        </p>
      )}

      {automaticLandmarksValid && (
        <p className="rounded-lg border border-lime/25 bg-lime/[0.05] px-3 py-2 text-sm text-lime">
          Stumps detected automatically. No manual correction required.
        </p>
      )}

      {needsManualCorrection && landmarks.length > 0 && (
        <p className="text-xs text-white/45">
          Adjust points only if a stump landmark is missing, low confidence, or registration
          failed after automatic detection.
        </p>
      )}

      {calibration?.warnings?.length ? (
        <ul className="space-y-1 text-xs text-white/50">
          {calibration.warnings.map((warning) => (
            <li key={warning}>• {warning}</li>
          ))}
        </ul>
      ) : null}

      {/* Uploaded footage is real cricket, so regulation geometry always
          applies. The custom-rig fields belong on /live, where an improvised
          indoor rig is the whole point; here they were only ever a way to get
          the pitch wrong. `pitchGeometry` stays null, which the solver reads
          as 22 yards. */}

      <div
        className="flex flex-wrap gap-2"
        onPointerDown={(event) => event.stopPropagation()}
      >
        <Button type="button" disabled={busy || phase === "accepted"} onClick={() => void runDetect()}>
          {phase === "boxes" ? "Detect & Register" : "Detect"}
        </Button>
        {needsManualCorrection && (
          <Button
            type="button"
            variant="secondary"
            disabled={busy || landmarks.length === 0 || phase === "accepted"}
            onClick={() => void runRegister()}
          >
            Adjust points & Register
          </Button>
        )}
        <Button
          type="button"
          variant="secondary"
          disabled={busy || !calibration || phase === "accepted"}
          onClick={() => void runAccept()}
        >
          Accept
        </Button>
        <Button type="button" variant="secondary" disabled={busy} onClick={resetBoxes}>
          Reset boxes
        </Button>
        {phase === "accepted" && onContinue && (
          <Button type="button" onClick={onContinue}>Continue</Button>
        )}
      </div>
    </Card>
  );
}
