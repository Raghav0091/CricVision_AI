"use client";

import Image from "next/image";
import { type ChangeEvent, useEffect, useRef, useState } from "react";

import { CalibrationStatus } from "@/components/live/CalibrationStatus";
import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { DeliveryCapturePanel } from "@/components/live/DeliveryCapturePanel";
import { DeliveryReview } from "@/components/live/DeliveryReview";
import { DraggableGuideBoxes } from "@/components/live/DraggableGuideBoxes";
import { DraggableStumpKeypoints, seedKeypointsFromBox } from "@/components/live/DraggableStumpKeypoints";
import { ExperimentalDeliveryTest } from "@/components/live/ExperimentalDeliveryTest";
import { InstructionCard } from "@/components/live/InstructionCard";
import { SetupWizard, shouldSkipSetupWizard } from "@/components/live/SetupWizard";
import { PitchSetupControl } from "@/components/live/PitchSetupControl";
import {
  CAMERA_ALIGNMENT_BOXES,
  type MappingLandmarks,
  StumpAlignmentOverlay,
  UPLOAD_ALIGNMENT_BOXES
} from "@/components/live/StumpAlignmentOverlay";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { LensCalibrationBadge } from "@/components/ui/LensCalibrationBadge";
import { solveCalibration, type ReplayPayloadV1 } from "@/lib/api";
import { runDeliveryAnalysis, type DeliveryProgress } from "@/lib/deliveryPipeline";
import { describePitchGeometry, loadPitchGeometry } from "@/lib/pitchGeometry";
import type { CricketPitchGeometry } from "@/lib/wicketCalibration/types";
import type {
  BoxLayout,
  CalibrationResponse,
  CapturedFrame,
  DetectionBoundingBox,
  KeypointLayout,
  LiveStage,
  NormalizedBox,
  PitchOverlay,
  PixelPoint,
  StumpDetection,
  StumpKeypointName,
  StumpKeypointSet,
  VirtualStumpEnd,
  VirtualStumpGeometry
} from "@/lib/types";


type LiveMode = "camera" | "upload" | "experimental";


const setupItems = [
  ["Fixed camera", "Use a tripod or stable phone mount."],
  ["Six stumps visible", "Keep both stump sets clearly inside the frame."],
  ["Behind non-striker", "Place the camera behind the non-striker stumps."],
  ["Good lighting", "Avoid shadows, glare, and low-light motion blur."],
  ["Clear view", "Keep players and equipment from blocking the camera."]
];


function readUploadedFrame(file: File): Promise<CapturedFrame> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read that image."));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("Could not read that image."));
        return;
      }
      const image = new window.Image();
      image.onerror = () => reject(new Error("Choose a valid JPG, PNG, or WebP image."));
      image.onload = () => resolve({ dataUrl: reader.result as string, width: image.naturalWidth, height: image.naturalHeight });
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}


function seedKeypointLayout(boxLayout: BoxLayout): KeypointLayout {
  return {
    striker: seedKeypointsFromBox(boxLayout.striker),
    non_striker: seedKeypointsFromBox(boxLayout.non_striker)
  };
}

// Real, user-placed points (not a bbox-fraction guess) — each dragged dot maps
// straight to pixel space, so this is the actual "manual_keypoints" geometry
// rather than the estimated_from_bbox interpolation used elsewhere.
function virtualStumpEndFromKeypoints(points: StumpKeypointSet, frameWidth: number, frameHeight: number): VirtualStumpEnd {
  const px = (point: { x: number; y: number }) => ({ x: Math.round(point.x * frameWidth), y: Math.round(point.y * frameHeight) });
  return {
    geometry_type: "manual_keypoints",
    stumps: [
      { name: "left", top: px(points.left_top), base: px(points.left_base) },
      { name: "middle", top: px(points.middle_top), base: px(points.middle_base) },
      { name: "right", top: px(points.right_top), base: px(points.right_base) }
    ],
    bails: [
      { name: "left_bail", start: px(points.left_top), end: px(points.middle_top) },
      { name: "right_bail", start: px(points.middle_top), end: px(points.right_top) }
    ]
  };
}

function bboxFromKeypoints(points: StumpKeypointSet, frameWidth: number, frameHeight: number): DetectionBoundingBox {
  const values = Object.values(points);
  const xs = values.map((point) => point.x * frameWidth);
  const tops = ["left_top", "middle_top", "right_top"] as const;
  const bases = ["left_base", "middle_base", "right_base"] as const;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...tops.map((key) => points[key].y * frameHeight));
  const maxY = Math.max(...bases.map((key) => points[key].y * frameHeight));
  return { x: Math.round(minX), y: Math.round(minY), width: Math.max(1, Math.round(maxX - minX)), height: Math.max(1, Math.round(maxY - minY)) };
}

// The corridor is the wicket-to-wicket ball path, so it spans the stump group
// and nothing more — a corridor drawn at full pitch width swamps the frame and
// buries the stumps it is supposed to sit behind.
// Length (non_striker <-> striker) is untouched; only the x-extent changes.
const PITCH_CORRIDOR_WIDEN_FACTOR = 1.0;
// The bowling crease is 2.64m (BOWLING_CREASE_LENGTH_M) and the detected stump
// bbox is one wicket wide (WICKET_WIDTH_M 0.2286), so the crease spans 11.55
// bbox widths. Source of truth: cricket_pitch_geometry.py.
const CREASE_WIDEN_FACTOR = 2.64 / 0.2286;

function buildPitchOverlay(
  strikerBbox: DetectionBoundingBox,
  nonStrikerBbox: DetectionBoundingBox,
  frameWidth: number,
  frameHeight: number,
  wickets: VirtualStumpGeometry
): PitchOverlay {
  const center = (box: DetectionBoundingBox) => ({
    x: Math.max(0, Math.min(frameWidth - 1, Math.round(box.x + box.width / 2))),
    y: Math.max(0, Math.min(frameHeight - 1, Math.round(box.y + box.height)))
  });
  const widenedEdges = (box: DetectionBoundingBox, factor: number): [number, number] => {
    const midX = box.x + box.width / 2;
    const half = (box.width * factor) / 2;
    return [Math.max(0, Math.round(midX - half)), Math.min(frameWidth - 1, Math.round(midX + half))];
  };
  const strikerCenter = center(strikerBbox);
  const nonStrikerCenter = center(nonStrikerBbox);
  const [nonStrikerLeft, nonStrikerRight] = widenedEdges(nonStrikerBbox, PITCH_CORRIDOR_WIDEN_FACTOR);
  const [strikerLeft, strikerRight] = widenedEdges(strikerBbox, PITCH_CORRIDOR_WIDEN_FACTOR);
  const [nonStrikerCreaseLeft, nonStrikerCreaseRight] = widenedEdges(nonStrikerBbox, CREASE_WIDEN_FACTOR);
  const [strikerCreaseLeft, strikerCreaseRight] = widenedEdges(strikerBbox, CREASE_WIDEN_FACTOR);
  const corridor = [
    { x: nonStrikerLeft, y: nonStrikerCenter.y },
    { x: strikerLeft, y: strikerCenter.y },
    { x: strikerRight, y: strikerCenter.y },
    { x: nonStrikerRight, y: nonStrikerCenter.y }
  ];
  const crease = (box: DetectionBoundingBox, left: number, right: number) => {
    const y = Math.min(frameHeight - 1, box.y + box.height);
    return [{ x: left, y }, { x: right, y }];
  };
  return {
    geometry_type: "estimated_from_stump_bboxes",
    pitch_axis: { start: nonStrikerCenter, end: strikerCenter },
    pitch_corridor: corridor,
    center_line: [nonStrikerCenter, strikerCenter],
    wickets,
    crease_guides: {
      striker: crease(strikerBbox, strikerCreaseLeft, strikerCreaseRight),
      non_striker: crease(nonStrikerBbox, nonStrikerCreaseLeft, nonStrikerCreaseRight)
    }
  };
}

function failureMessage(result: CalibrationResponse, mode: LiveMode): string {
  if (result.status === "stump_detector_missing") {
    return "Stump detector model is missing. Add Models/stump_detector/best.pt.";
  }
  if (result.status === "stumps_not_found") {
    if (mode === "upload") {
      const strikerFound = result.detections?.striker.found === true;
      const nonStrikerFound = result.detections?.non_striker.found === true;
      if (nonStrikerFound && !strikerFound) {
        return "Non-striker stumps detected. Move the striker box over the far stumps and press Redetect.";
      }
      if (strikerFound && !nonStrikerFound) {
        return "Striker stumps detected. Move the non-striker box over the near stumps and press Redetect.";
      }
      return "Stumps not found. Adjust both boxes or upload a clearer image.";
    }
    return "Place real cricket stumps inside both boxes and try again.";
  }
  return result.message;
}


function BoxControls({
  layout,
  onChange,
  onReset
}: {
  layout: BoxLayout;
  onChange: (end: keyof BoxLayout, field: keyof NormalizedBox, value: number) => void;
  onReset: () => void;
}) {
  const controls: Array<{ field: keyof NormalizedBox; label: string; min: number; max: number }> = [
    { field: "x", label: "X", min: 0, max: 96 },
    { field: "y", label: "Y", min: 0, max: 92 },
    { field: "width", label: "Width", min: 4, max: 50 },
    { field: "height", label: "Height", min: 8, max: 50 }
  ];
  return (
    <div className="mt-5 rounded-xl border border-white/10 bg-black/20 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/45">Detection boxes</p>
        <button type="button" className="text-xs font-bold text-lime hover:text-white" onClick={onReset}>Reset defaults</button>
      </div>
      {(["striker", "non_striker"] as const).map((end) => (
        <div key={end} className="mt-4">
          <p className="text-xs font-bold capitalize text-white/70">{end.replace("_", " ")}</p>
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-3">
            {controls.map(({ field, label, min, max }) => (
              <label key={`${end}-${field}`} className="text-[11px] text-white/45">
                <span className="flex justify-between gap-2"><span>{label}</span><span>{Math.round(layout[end][field] * 100)}%</span></span>
                <input
                  className="mt-1 h-1.5 w-full cursor-pointer accent-lime"
                  type="range"
                  min={min}
                  max={max}
                  step="1"
                  value={Math.round(layout[end][field] * 100)}
                  onChange={(event) => onChange(end, field, Number(event.target.value) / 100)}
                />
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}


// Deviation from vertical. `beta` is 90 with the phone upright and 0 laid
// flat, so a phone being raised walks this value down toward the threshold.
const UPRIGHT_THRESHOLD_DEG = 10;

function tiltFromVertical(beta: number | null): number | null {
  if (beta === null || Number.isNaN(beta)) return null;
  return Math.min(180, Math.max(0, Math.round(Math.abs(90 - beta))));
}

type OrientationPermission = "granted" | "unavailable";

// iOS 13+ gates DeviceOrientationEvent behind a user gesture, so this must be
// called from the wizard's Start press rather than from an effect.
async function requestOrientationPermission(): Promise<OrientationPermission> {
  if (typeof window === "undefined" || typeof DeviceOrientationEvent === "undefined") return "unavailable";
  const requestable = DeviceOrientationEvent as unknown as { requestPermission?: () => Promise<string> };
  if (typeof requestable.requestPermission !== "function") return "granted";
  try {
    return (await requestable.requestPermission()) === "granted" ? "granted" : "unavailable";
  } catch {
    return "unavailable";
  }
}

// The release point is clipped when the far (striker) detection is jammed
// against the top of frame — the bowler's arm is then outside the capture.
const RELEASE_CLIP_FRACTION = 0.08;

function releasePointClipped(result: CalibrationResponse, frameHeight: number): boolean {
  const bbox = result.detections?.striker.bbox;
  if (!bbox || !frameHeight) return false;
  return bbox.y <= frameHeight * RELEASE_CLIP_FRACTION;
}

function landmarksFromResult(result: CalibrationResponse | null): PixelPoint[] {
  if (!result?.virtual_stumps) return [];
  return (["striker", "non_striker"] as const).flatMap((end) =>
    (result.virtual_stumps?.[end]?.stumps ?? []).flatMap((stump) => [stump.top, stump.base])
  );
}


function TiltCameraUpModal({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="absolute inset-0 z-10 grid place-items-center bg-black/50 p-6">
      <div className="w-full max-w-[20rem] rounded-2xl bg-white p-5 text-black shadow-xl">
        <h2 className="text-lg font-bold">Tilt the Camera Up</h2>
        <p className="mt-2 text-sm leading-6 text-black/70">
          The bowler&rsquo;s release point is cut off at the top. Please point the camera up slightly.
        </p>
        <div className="mt-4 flex justify-end">
          <button type="button" className="px-2 py-1 text-sm font-bold text-[#1A73E8]" onClick={onDismiss}>OK</button>
        </div>
      </div>
    </div>
  );
}


function DetectionSummary({ result }: { result: CalibrationResponse | null }) {
  if (!result?.detections) return null;
  return (
    <div className="mt-5 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/45">Detector evidence</p>
        {result.virtual_stumps && <span className="text-right text-[10px] font-bold uppercase text-[#ffe761]">Estimated from bbox</span>}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        {(["striker", "non_striker"] as const).map((end) => {
          const detection = result.detections?.[end];
          return (
            <div key={end} className="rounded-lg bg-black/20 p-3">
              <p className="text-xs capitalize text-white/45">{end.replace("_", " ")}</p>
              <p className={`mt-1 text-sm font-bold ${detection?.found ? "text-lime" : "text-[#ffaaa6]"}`}>
                {detection?.found ? `${(detection.confidence * 100).toFixed(1)}%` : "Not found"}
              </p>
            </div>
          );
        })}
      </div>
      {result.calibration_quality && (
        <p className="mt-3 text-xs text-white/45">Calibration score {(result.calibration_quality.score * 100).toFixed(1)}%</p>
      )}
    </div>
  );
}


export default function LivePage() {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const [mode, setMode] = useState<LiveMode>("camera");
  const [stage, setStage] = useState<LiveStage>("setup");
  const [message, setMessage] = useState<string | null>(null);
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResponse | null>(null);
  const [frameSize, setFrameSize] = useState<{ width: number; height: number } | null>(null);
  const [uploadedFrame, setUploadedFrame] = useState<CapturedFrame | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadBoxLayout, setUploadBoxLayout] = useState<BoxLayout>(UPLOAD_ALIGNMENT_BOXES);
  const [deliveryCount] = useState(0);
  const [manualKeypointMode, setManualKeypointMode] = useState(false);
  const [manualKeypoints, setManualKeypoints] = useState<KeypointLayout>(() => seedKeypointLayout(CAMERA_ALIGNMENT_BOXES));
  const [showWizard, setShowWizard] = useState(false);
  // Null means the regulation 22 yards, which is the normal case. Read from
  // localStorage on mount so a test rig survives a reload.
  const [pitchGeometry, setPitchGeometry] = useState<CricketPitchGeometry | null>(null);
  // One-delivery capture: record from the already-open camera stream, then run
  // the same chain /video-analysis uses, because that is the only path that
  // yields a metric camera pose and therefore real speed/length numbers.
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const [recording, setRecording] = useState(false);
  const [deliveryProgress, setDeliveryProgress] = useState<DeliveryProgress | null>(null);
  const [deliveryError, setDeliveryError] = useState<string | null>(null);
  const [deliveryClipUrl, setDeliveryClipUrl] = useState<string | null>(null);
  const [deliveryReplay, setDeliveryReplay] = useState<ReplayPayloadV1 | null>(null);

  // Object URLs outlive the component unless revoked explicitly.
  useEffect(() => () => {
    if (deliveryClipUrl) URL.revokeObjectURL(deliveryClipUrl);
  }, [deliveryClipUrl]);

  function startDeliveryRecording() {
    const stream = cameraRef.current?.getStream();
    if (!stream) {
      setDeliveryError("Camera stream is not ready.");
      return;
    }
    if (typeof window.MediaRecorder === "undefined") {
      setDeliveryError("This browser cannot record video. Try current Chrome, Edge, or Safari.");
      return;
    }
    const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"]
      .find((candidate) => MediaRecorder.isTypeSupported(candidate));
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    chunksRef.current = [];
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      recorderRef.current = null;
      const type = recorder.mimeType || chunksRef.current[0]?.type || "video/webm";
      const clip = new Blob(chunksRef.current, { type });
      chunksRef.current = [];
      if (clip.size === 0) {
        setDeliveryError("No playable clip was recorded.");
        return;
      }
      void analyseDelivery(clip);
    };
    setDeliveryError(null);
    setDeliveryReplay(null);
    recorder.start(250);
    setRecording(true);
  }

  function stopDeliveryRecording() {
    setRecording(false);
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  // `manualKeypoints` starts as seeded defaults derived from the alignment box,
  // so it is always populated whether or not anyone touched it. Sending those
  // to the solver asks it to explain a rectangle nobody placed — it reports a
  // large reprojection error and every physical check passes, because a perfect
  // rectangle is perfectly self-consistent. Only real placements should go.
  function hasPlacedKeypoints(): boolean {
    if (!manualKeypointMode) return false;
    const seed = seedKeypointLayout(CAMERA_ALIGNMENT_BOXES);
    return (["striker", "non_striker"] as const).some((end) =>
      (Object.keys(manualKeypoints[end]) as StumpKeypointName[]).some((name) => {
        const placed = manualKeypoints[end][name];
        const seeded = seed[end][name];
        return Math.abs(placed.x - seeded.x) > 0.002 || Math.abs(placed.y - seeded.y) > 0.002;
      })
    );
  }

  // Map a whole keypoint layout out of preview space and into frame space.
  // Falls back to the original points when the preview cannot report its
  // geometry, which is no worse than the previous behaviour.
  function toFrameKeypoints(layout: KeypointLayout): KeypointLayout {
    const convert = cameraRef.current?.frameFromPreviewPoint;
    if (!convert) return layout;
    const mapSet = (set: StumpKeypointSet): StumpKeypointSet => {
      const next = { ...set };
      for (const name of Object.keys(set) as StumpKeypointName[]) {
        const mapped = cameraRef.current?.frameFromPreviewPoint(set[name]);
        if (mapped) next[name] = mapped;
      }
      return next;
    };
    return { striker: mapSet(layout.striker), non_striker: mapSet(layout.non_striker) };
  }

  async function analyseDelivery(clip: Blob) {
    setDeliveryClipUrl(URL.createObjectURL(clip));
    try {
      // The boxes and points the operator actually aligned, not the defaults —
      // otherwise the pipeline re-detects from scratch and loses the manual
      // placement that made calibration succeed in the first place.
      //
      // Points are placed against the preview box, which is object-cover, so a
      // tall frame shows only its centre crop. Sending preview coordinates
      // straight to the solver puts every stump point hundreds of pixels from
      // where it was placed.
      const result = await runDeliveryAnalysis(
        clip,
        mode === "upload" ? uploadBoxLayout : CAMERA_ALIGNMENT_BOXES,
        setDeliveryProgress,
        hasPlacedKeypoints()
          ? (mode === "camera" ? toFrameKeypoints(manualKeypoints) : manualKeypoints)
          : null,
        pitchGeometry
      );
      setDeliveryReplay(result.replay);
    } catch (error) {
      setDeliveryError(error instanceof Error ? error.message : "Could not analyse that delivery.");
    } finally {
      setDeliveryProgress(null);
    }
  }

  function retakeDelivery() {
    if (deliveryClipUrl) URL.revokeObjectURL(deliveryClipUrl);
    setDeliveryClipUrl(null);
    setDeliveryReplay(null);
    setDeliveryError(null);
    setStage("capturing");
  }
  const [tiltAngle, setTiltAngle] = useState<number | null>(null);
  const [orientationUnavailable, setOrientationUnavailable] = useState(false);
  const [cameraBoxLayout, setCameraBoxLayout] = useState<BoxLayout>(CAMERA_ALIGNMENT_BOXES);
  const [manualBoxMode, setManualBoxMode] = useState(false);
  const [mappingLandmarks, setMappingLandmarks] = useState<MappingLandmarks | null>(null);
  const [showTiltUp, setShowTiltUp] = useState(false);

  // The gate reads live tilt only while it is on screen; leaving it stops the
  // listener so the sensor is not driving renders during capture.
  useEffect(() => {
    if (stage !== "orientation-gate" || orientationUnavailable) return;
    function onOrientation(event: DeviceOrientationEvent) {
      setTiltAngle(tiltFromVertical(event.beta));
    }
    window.addEventListener("deviceorientation", onOrientation);
    return () => window.removeEventListener("deviceorientation", onOrientation);
  }, [stage, orientationUnavailable]);

  // Never trap the user behind a sensor that does not exist: no reading at all
  // after a moment, or a reading already inside tolerance, moves the flow on.
  useEffect(() => {
    if (stage !== "orientation-gate") return;
    if (orientationUnavailable || (tiltAngle !== null && tiltAngle < UPRIGHT_THRESHOLD_DEG)) {
      setStage("align-stumps");
      return;
    }
    if (tiltAngle !== null) return;
    const timer = window.setTimeout(() => setOrientationUnavailable(true), 1500);
    return () => window.clearTimeout(timer);
  }, [stage, tiltAngle, orientationUnavailable]);

  // localStorage is only readable in the browser, so the declared pitch has to
  // be picked up after mount rather than in the initial state.
  useEffect(() => {
    setPitchGeometry(loadPitchGeometry());
  }, []);

  // Camera capture is the only mode the guidelines apply to — upload and the
  // experimental test never involve placing a tripod.
  function startCameraCalibration() {
    if (shouldSkipSetupWizard()) {
      void startCameraFromGesture();
      return;
    }
    setShowWizard(true);
  }

  // The wizard's Start press is the user gesture iOS requires for the
  // orientation permission prompt.
  async function startCameraFromGesture() {
    const permission = await requestOrientationPermission();
    setOrientationUnavailable(permission === "unavailable");
    setTiltAngle(null);
    enterCalibration("camera");
  }

  function enterCalibration(nextMode: LiveMode) {
    setShowWizard(false);
    setMode(nextMode);
    setMessage(null);
    setCalibrationResult(null);
    setMappingLandmarks(null);
    setShowTiltUp(false);
    setManualKeypointMode(false);
    setManualBoxMode(false);
    setCameraBoxLayout(CAMERA_ALIGNMENT_BOXES);
    setManualKeypoints(seedKeypointLayout(nextMode === "upload" ? UPLOAD_ALIGNMENT_BOXES : CAMERA_ALIGNMENT_BOXES));
    setFrameSize(nextMode === "upload" && uploadedFrame ? { width: uploadedFrame.width, height: uploadedFrame.height } : null);
    setStage(nextMode === "camera" ? "orientation-gate" : "align-stumps");
  }

  function confirmManualKeypoints() {
    const frame = mode === "camera" ? cameraRef.current?.captureFrame() : uploadedFrame;
    if (!frame) {
      setMessage(mode === "upload" ? "Upload a cricket image to test stump calibration." : "Camera frame is not ready. Wait a moment and try again.");
      return;
    }
    setMessage(null);
    setFrameSize({ width: frame.width, height: frame.height });

    const virtualStumps: VirtualStumpGeometry = {
      striker: virtualStumpEndFromKeypoints(manualKeypoints.striker, frame.width, frame.height),
      non_striker: virtualStumpEndFromKeypoints(manualKeypoints.non_striker, frame.width, frame.height)
    };
    const strikerBbox = bboxFromKeypoints(manualKeypoints.striker, frame.width, frame.height);
    const nonStrikerBbox = bboxFromKeypoints(manualKeypoints.non_striker, frame.width, frame.height);
    const pitchOverlay = buildPitchOverlay(strikerBbox, nonStrikerBbox, frame.width, frame.height, virtualStumps);
    const detections: Record<"striker" | "non_striker", StumpDetection> = {
      striker: { found: true, confidence: 1, bbox: strikerBbox, source_box: "striker", class_name: "manual_keypoints" },
      non_striker: { found: true, confidence: 1, bbox: nonStrikerBbox, source_box: "non_striker", class_name: "manual_keypoints" }
    };

    setCalibrationResult({
      success: true,
      status: "setup_complete",
      quality: "Unavailable",
      reason: "manual_keypoints",
      message: "Using your six manually placed points per end as the virtual stumps — no ML detection was run.",
      detections,
      virtual_stumps: virtualStumps,
      pitch_overlay: pitchOverlay
    });
    setStage("setup-complete");
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const frame = await readUploadedFrame(file);
      setUploadedFrame(frame);
      setUploadedFileName(file.name);
      setUploadBoxLayout(UPLOAD_ALIGNMENT_BOXES);
      setManualKeypoints(seedKeypointLayout(UPLOAD_ALIGNMENT_BOXES));
      setFrameSize({ width: frame.width, height: frame.height });
      setCalibrationResult(null);
      setMessage(null);
      setStage("align-stumps");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not read that image.");
    } finally {
      event.target.value = "";
    }
  }

  function updateUploadBox(end: keyof BoxLayout, field: keyof NormalizedBox, value: number) {
    setUploadBoxLayout((current) => {
      const box = current[end];
      const next = { ...box, [field]: value };
      if (field === "x") next.x = Math.min(value, 1 - box.width);
      if (field === "y") next.y = Math.min(value, 1 - box.height);
      if (field === "width") next.width = Math.min(value, 1 - box.x);
      if (field === "height") next.height = Math.min(value, 1 - box.y);
      return { ...current, [end]: next };
    });
  }

  async function continueCalibration() {
    const frame = mode === "camera" ? cameraRef.current?.captureFrame() : uploadedFrame;
    if (!frame) {
      setMessage(mode === "upload" ? "Upload a cricket image to test stump calibration." : "Camera frame is not ready. Wait a moment and try again.");
      return;
    }
    setFrameSize({ width: frame.width, height: frame.height });
    setMessage(null);
    setCalibrationResult(null);
    setMappingLandmarks(null);
    setStage(mode === "camera" ? "mapping" : "solving-calibration");
    try {
      const boxLayout = mode === "upload" ? uploadBoxLayout : cameraBoxLayout;
      const result = await solveCalibration(frame, boxLayout);
      const landmarks = landmarksFromResult(result);
      setMappingLandmarks({ raw: landmarks, accepted: landmarks });

      // A single-wicket solve is degenerate: without the far wicket there is no
      // pitch axis, so it must never reach setup-complete.
      const bothStumpSetsFound = result.detections?.striker.found === true && result.detections?.non_striker.found === true;
      if (!result.success || !bothStumpSetsFound) {
        setCalibrationResult(result);
        setMessage(result.status === "setup_complete" ? "Stumps not found. Adjust both boxes or upload a clearer image." : failureMessage(result, mode));
        setStage("align-stumps");
        return;
      }
      if (mode === "camera" && releasePointClipped(result, frame.height)) {
        setShowTiltUp(true);
        return;
      }
      // Freeze the solved pose. Nothing re-solves per frame — a locked anchor
      // reads as stable even when slightly off, where a live one jitters.
      setCalibrationResult(result);
      setStage("setup-complete");
    } catch (error) {
      setMessage(error instanceof Error && error.message !== "Failed to fetch" ? error.message : "Calibration backend is unavailable. Confirm FastAPI is running at the configured API URL.");
      setStage("align-stumps");
    }
  }

  function redetect() {
    setCalibrationResult(null);
    setMappingLandmarks(null);
    setMessage(null);
    setStage("align-stumps");
  }

  if (stage === "setup" && showWizard) {
    return (
      <div className="py-5">
        <SetupWizard onComplete={() => void startCameraFromGesture()} onBack={() => setShowWizard(false)} />
      </div>
    );
  }

  if (stage === "setup") {
    return (
      <div className="mx-auto max-w-6xl py-5">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge label="Live bowling session" tone="good" />
          <LensCalibrationBadge />
        </div>
        <div className="mt-6 grid gap-8 lg:grid-cols-[1.25fr_0.75fr] lg:items-end">
          <div>
            <h1 className="text-5xl font-black tracking-[-0.05em] sm:text-7xl">Set the pitch.<br /><span className="text-lime">Test the calibration.</span></h1>
            <p className="mt-6 max-w-xl text-base leading-7 text-white/55">Use the live camera or upload a cricket image. CricVision only completes setup after both stump sets are detected.</p>
          </div>
          <div className="grid gap-3">
            {/* Declared before calibration, because registration solves against
                whatever pitch it is told about. */}
            <PitchSetupControl geometry={pitchGeometry} onChange={setPitchGeometry} />
            <Button className="w-full py-4 text-base" onClick={startCameraCalibration}>Camera Calibration</Button>
            <Button className="w-full py-4 text-base" variant="secondary" onClick={() => enterCalibration("upload")}>Upload Calibration Image</Button>
            <Button className="w-full py-4 text-base" variant="secondary" onClick={() => enterCalibration("experimental")}>Experimental Delivery Test</Button>
          </div>
        </div>
        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {setupItems.map(([title, copy], index) => (
            <Card key={title} className="shadow-none">
              <span className="text-xs font-black text-lime">0{index + 1}</span>
              <h2 className="mt-8 font-bold">{title}</h2>
              <p className="mt-2 text-sm leading-5 text-white/45">{copy}</p>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  // A finished delivery takes over the whole screen, the way FullTrack's
  // review does — the camera stages behind it are irrelevant at that point.
  if (deliveryReplay && deliveryClipUrl) {
    return (
      <div className="fixed inset-0 z-50">
        <DeliveryReview
          clipUrl={deliveryClipUrl}
          replay={deliveryReplay}
          onRetake={retakeDelivery}
          onClose={() => {
            retakeDelivery();
            setStage("setup");
          }}
          pitchGeometry={pitchGeometry}
        />
      </div>
    );
  }

  if (mode === "experimental") {
    return <ExperimentalDeliveryTest onSelectCalibration={(nextMode) => enterCalibration(nextMode)} />;
  }

  // Full-screen, mobile-app-style shell for every calibration/capture stage in
  // both Camera and Upload modes — a split half-screen layout hides too much
  // of the frame to place six keypoints per end accurately.
  if (stage !== "results") {
    // Boxes are hidden while the orientation warning is up and while mapping
    // runs, so the user's attention is on the card or the landmark dots.
    const showAlignmentBoxes = stage === "align-stumps" || stage === "solving-calibration";
    const showOrientationWarning = stage === "orientation-gate" && !orientationUnavailable && tiltAngle !== null;
    return (
      <div className="fixed inset-0 z-50 flex flex-col bg-black">
        <div className="relative flex-1 overflow-hidden">
          {mode === "camera" ? (
            <CameraPreview ref={cameraRef} className="absolute inset-0 h-full w-full bg-black" />
          ) : uploadedFrame ? (
            <Image src={uploadedFrame.dataUrl} alt="Uploaded cricket calibration" fill unoptimized className="absolute inset-0 object-contain" />
          ) : (
            <div className="absolute inset-0 grid place-items-center bg-ink p-8 text-center">
              <div>
                <p className="text-lg font-bold">Upload a cricket image to test stump calibration.</p>
                <p className="mt-2 text-sm text-white/45">Use a clear JPG, PNG, or WebP showing both stump sets.</p>
                <label htmlFor="calibration-image" className="mt-5 inline-block cursor-pointer rounded-xl border border-dashed border-white/20 bg-white/5 px-5 py-3 text-sm font-bold transition hover:bg-white/10">
                  Choose calibration image
                </label>
                <input id="calibration-image" className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => void handleUpload(event)} />
              </div>
            </div>
          )}
          {stage === "align-stumps" && manualKeypointMode ? (
            <DraggableStumpKeypoints layout={manualKeypoints} onChange={setManualKeypoints} editable />
          ) : stage === "align-stumps" && manualBoxMode ? (
            <DraggableGuideBoxes layout={cameraBoxLayout} onChange={setCameraBoxLayout} editable />
          ) : (
            <StumpAlignmentOverlay
              showAlignment={showAlignmentBoxes && !showOrientationWarning}
              detections={calibrationResult?.detections}
              virtualStumps={calibrationResult?.virtual_stumps}
              pitchOverlay={calibrationResult?.pitch_overlay}
              boxLayout={mode === "upload" ? uploadBoxLayout : cameraBoxLayout}
              frameWidth={frameSize?.width}
              frameHeight={frameSize?.height}
              mappingLandmarks={stage === "mapping" ? mappingLandmarks : null}
              setupComplete={stage === "setup-complete" && calibrationResult?.success === true}
            />
          )}
          {stage === "capturing" && <div className="absolute left-4 top-4 rounded-full bg-signal px-3 py-1 text-xs font-black uppercase tracking-wider">Live</div>}
          {stage === "setup-complete" && (
            <button
              type="button"
              className="absolute left-4 top-4 rounded-full bg-white px-4 py-2 text-xs font-bold uppercase tracking-wide text-[#1A73E8]"
              onClick={redetect}
            >
              Redetect
            </button>
          )}
          <button
            type="button"
            aria-label="Exit calibration"
            className="absolute right-4 top-4 grid h-8 w-8 place-items-center rounded-full bg-white/85 text-lg font-bold leading-none text-ink"
            onClick={() => { setMessage(null); setCalibrationResult(null); setStage("setup"); }}
          >
            ×
          </button>

          {/* The operator must be able to see, while capturing, that this is a
              declared rig and not a full-length pitch. */}
          {describePitchGeometry(pitchGeometry) && (
            <div className="absolute inset-x-0 top-14 mx-auto w-fit rounded-full border border-[#ffcf4d]/40 bg-black/70 px-3 py-1 text-[11px] font-semibold text-[#ffcf4d] backdrop-blur-sm">
              Custom rig · {describePitchGeometry(pitchGeometry)}
            </div>
          )}

          {showOrientationWarning && (
            <InstructionCard tone="warning">
              Please hold the device in portrait view<br />
              Current Angle: {tiltAngle}°<br />
              It should be less than {UPRIGHT_THRESHOLD_DEG}°
            </InstructionCard>
          )}
          {mode === "camera" && stage === "align-stumps" && !manualKeypointMode && (
            <InstructionCard>
              Fit the stumps completely in the boxes then press Continue.<br />
              Pinch to zoom in or out!
            </InstructionCard>
          )}
          {stage === "mapping" && (
            <InstructionCard>
              Please wait while we map the stumps.<br />
              Make sure the non-striker stumps are fully visible.
            </InstructionCard>
          )}
          {stage === "setup-complete" && calibrationResult?.success === true && (
            <InstructionCard>
              Setup Complete. Press &ldquo;Redetect&rdquo; if the pitch is not detected correctly.<br />
              Start bowling to trigger the recording!
            </InstructionCard>
          )}

          {mode === "camera" && stage === "align-stumps" && !manualKeypointMode && !manualBoxMode && (
            <Button
              variant="camera"
              className="absolute bottom-8 left-1/2 -translate-x-1/2"
              onClick={() => void continueCalibration()}
            >
              Continue
            </Button>
          )}
          {showTiltUp && <TiltCameraUpModal onDismiss={() => { setShowTiltUp(false); redetect(); }} />}
        </div>
        <div
          className={`max-h-[60vh] shrink-0 overflow-y-auto border-t border-white/10 bg-ink/95 p-4 backdrop-blur ${stage === "orientation-gate" ? "hidden" : ""}`}
        >
          {stage === "align-stumps" && (
            <>
              <CalibrationStatus
                status={message ? "Failed" : "Searching"}
                message={
                  message
                  ?? (manualKeypointMode
                    ? "Drag each of the six points per end onto the stump tops and bases, then press Use these keypoints."
                    : mode === "upload"
                      ? (uploadedFrame ? "Press Detect to find both stump sets and build the pitch overlay." : "Upload a cricket image to test stump calibration.")
                      : "Move the camera so both stump sets sit inside the red boxes, then press Continue.")
                }
              />
              {/* The points start as a seeded rectangle, so this screen looks
                  finished before anything has actually been measured. Saying so
                  is the difference between a good solve and a 250px reprojection
                  error surfaced three screens later. */}
              {manualKeypointMode && !hasPlacedKeypoints() && (
                <p className="rounded-xl border border-[#ffcf4d]/30 bg-[#ffcf4d]/10 p-3 text-sm leading-6 text-[#ffcf4d]">
                  These points are still in their default positions. Drag all 12 onto the real stumps —
                  the tip and the base of each one — or nothing gets measured. Pinch to zoom in for the
                  far end.
                </p>
              )}
              {!manualKeypointMode && <DetectionSummary result={calibrationResult} />}
              {!manualKeypointMode && mode === "upload" && uploadedFrame && (
                <BoxControls
                  layout={uploadBoxLayout}
                  onChange={updateUploadBox}
                  onReset={() => setUploadBoxLayout(UPLOAD_ALIGNMENT_BOXES)}
                />
              )}
              {/* The reference app has no manual fallback at all; keeping both
                  behind a text button is a better answer when detection
                  struggles, without letting the boxes drift by default. */}
              <div className="mt-3 flex flex-wrap gap-4">
                {mode === "camera" && !manualKeypointMode && (
                  <button
                    type="button"
                    className="text-xs font-bold text-lime hover:text-white"
                    onClick={() => setManualBoxMode((current) => !current)}
                  >
                    {manualBoxMode ? "Use fixed guide boxes" : "Adjust manually"}
                  </button>
                )}
                <button
                  type="button"
                  className="text-xs font-bold text-lime hover:text-white"
                  onClick={() => { setMessage(null); setCalibrationResult(null); setManualBoxMode(false); setManualKeypointMode((current) => !current); }}
                >
                  {manualKeypointMode ? "Use auto-detection instead" : "Skip detection — place all six stump keypoints myself"}
                </button>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <Button variant="secondary" onClick={() => { setMessage(null); setCalibrationResult(null); setStage("setup"); }}>Cancel</Button>
                {manualKeypointMode ? (
                  // Accepting the seeded rectangle produces a confident,
                  // completely wrong calibration, so it is not offered.
                  <Button disabled={!hasPlacedKeypoints()} onClick={confirmManualKeypoints}>
                    {hasPlacedKeypoints() ? "Use these keypoints" : "Drag the 12 points first"}
                  </Button>
                ) : mode === "upload" ? (
                  <Button disabled={!uploadedFrame} onClick={() => void continueCalibration()}>
                    {calibrationResult ? "Redetect" : "Detect"}
                  </Button>
                ) : (
                  // Camera mode drives Continue from the floating pill over the
                  // frame, so the panel would otherwise show it twice.
                  manualBoxMode && <Button onClick={() => void continueCalibration()}>Continue</Button>
                )}
              </div>
            </>
          )}
          {(stage === "solving-calibration" || stage === "mapping") && (
            <>
              <CalibrationStatus status="Solving" message="Checking stumps..." />
              <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full w-1/2 animate-pulse rounded-full bg-lime" /></div>
            </>
          )}
          {stage === "setup-complete" && (
            <>
              <CalibrationStatus status="Setup Complete" message="Both stump sets are mapped. Redetect is available top-left if the pitch looks wrong." />
              <DetectionSummary result={calibrationResult} />
              <Button className="mt-4 w-full" onClick={() => setStage("capturing")}>Start Capture</Button>
            </>
          )}
          {stage === "capturing" && (
            <>
              <DeliveryCapturePanel deliveryCount={deliveryCount} recording={recording} />
              {deliveryProgress ? (
                <Card className="mt-4">
                  <p className="text-sm font-bold">{deliveryProgress.message}</p>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-lime transition-[width]"
                      style={{ width: `${Math.max(6, Math.min(100, deliveryProgress.percent ?? 12))}%` }}
                    />
                  </div>
                </Card>
              ) : (
                <Button
                  className="mt-4 w-full"
                  variant={recording ? "danger" : "primary"}
                  onClick={recording ? stopDeliveryRecording : startDeliveryRecording}
                >
                  {recording ? "Stop and analyse" : "Record delivery"}
                </Button>
              )}
              {deliveryError && (
                <p className="mt-3 text-sm leading-5 text-[#ffaaa6]">{deliveryError}</p>
              )}
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl py-10 text-center">
      <Card>
        <CalibrationStatus status="Failed" message="No analysed delivery result is available yet." />
        <Button className="mt-6 w-full" variant="secondary" onClick={() => setStage("capturing")}>Return to Capture</Button>
      </Card>
    </div>
  );
}
