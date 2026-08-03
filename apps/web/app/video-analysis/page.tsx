"use client";

import dynamic from "next/dynamic";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
  type ChangeEvent,
  type DragEvent
} from "react";

import { Button } from "@/components/ui/Button";
import {
  CalibrationCanvas,
  DEFAULT_VIDEO_GUIDES
} from "@/components/video-analysis/CalibrationCanvas";
import { BallAnalysisReviewPanel } from "@/components/video-analysis/BallAnalysisReviewPanel";
import {
  BallAnalysisTimeline,
  buildTimelineMarkers
} from "@/components/video-analysis/BallAnalysisTimeline";
import { BallTrackOverlay } from "@/components/video-analysis/BallTrackOverlay";
import type { VirtualPitchSceneProps } from "@/components/virtual-pitch";
import {
  confirmVideoAnalysisCalibration,
  detectVideoAnalysisCalibration,
  fitConfirmedWicketCamera,
  getBallDetectorModels,
  getCameraSetupPresets,
  getVideoBallDetectionJob,
  getVideoBallDetectionResult,
  getVideoBallTrackingJob,
  getVideoBallTrackingResult,
  getVirtualPitchSpecification,
  prepareVideoAnalysis,
  startVideoBallDetection,
  startVideoBallTracking,
  type ConfirmedVideoCalibrationResponse,
  type BallDetectorModelKey,
  type BallDetectorModelOption,
  type NormalizedCameraBridgeResponse,
  type NormalizedBox,
  type VideoAnalysisPreparedResponse,
  type VideoBallDetectionJobStatus,
  type VideoBallDetectionResultResponse,
  type VideoBallDetectionSummary,
  type VideoBallTrackingJobStatus,
  type VideoBallTrackingResultResponse,
  type VideoCalibrationDetectionResponse,
  type WicketCalibration
} from "@/lib/api";
import {
  activeFrameCandidates as candidatesForFrame,
  activeFramePrimaryPoint,
  buildReviewCandidates,
  DEFAULT_BALL_REVIEW_TOGGLES,
  type BallReviewDisplayToggles
} from "@/lib/ball-analysis-review";
import {
  adaptVirtualPitchResponse,
  calculateCameraPreset,
  materialPreset,
  type VirtualPitchModel
} from "@/lib/virtual-pitch";

const VirtualPitchCanvas = dynamic<VirtualPitchSceneProps>(
  () => import("@/components/virtual-pitch/VirtualPitchCanvas").then(
    (module) => module.VirtualPitchCanvas as ComponentType<VirtualPitchSceneProps>
  ),
  { ssr: false }
);

type BoxSource = WicketCalibration["source"];

const FAR_DEFAULT: NormalizedBox = DEFAULT_VIDEO_GUIDES.striker;
const NEAR_DEFAULT: NormalizedBox = DEFAULT_VIDEO_GUIDES.non_striker;
const POLL_INTERVAL_MS = 1000;

type BallAnalysisPhase =
  | "idle"
  | "preparing_detector"
  | "detecting"
  | "filtering_static"
  | "selecting_candidates"
  | "tracking"
  | "recovering_gaps"
  | "scoring_primary"
  | "preparing_review"
  | "ready"
  | "failed";

const BALL_PROGRESS_LABELS: Record<BallAnalysisPhase, string> = {
  idle: "",
  preparing_detector: "Preparing detector...",
  detecting: "Detecting candidates...",
  filtering_static: "Filtering static candidates...",
  selecting_candidates: "Selecting moving-ball candidates...",
  tracking: "Building primary track...",
  recovering_gaps: "Recovering short gaps...",
  scoring_primary: "Scoring primary track...",
  preparing_review: "Preparing analysis review...",
  ready: "Analysis ready",
  failed: "Ball tracking unavailable"
};

const FIT_STATUS_LABELS: Record<NonNullable<NormalizedCameraBridgeResponse["fitStatus"]>, string> = {
  FIT_READY: "Pitch Fit Ready",
  FIT_APPROXIMATE: "Approximate Pitch Fit",
  FIT_FAILED: "Pitch Fit Failed"
};

function fitFailureMessage(bridge: NormalizedCameraBridgeResponse): string {
  const reasons = bridge.fitValidation?.reasons.filter(
    (reason) => reason !== "PROJECTED_WICKETS_REQUIRE_VISUAL_REVIEW"
  ) ?? [];
  if (reasons.length === 0) {
    return "Pitch fit failed due to invalid geometry. Adjust the boxes and refit.";
  }
  return `Pitch fit failed: ${reasons.join(", ")}. Adjust the boxes and refit.`;
}

function resolvedDetectorLabel(
  summary: VideoBallDetectionSummary | null,
  models: BallDetectorModelOption[]
): string | null {
  if (!summary?.detector) return null;
  const { requested_key: requestedKey, selected_key: selectedKey, display_name: displayName } = summary.detector;
  if (requestedKey === "automatic") {
    return `Automatic → ${selectedKey.toUpperCase()}`;
  }
  const option = models.find((model) => model.key === requestedKey);
  return option?.display_name ?? displayName;
}

function persistedResultsMatch(
  detectionResult: VideoBallDetectionResultResponse | null,
  trackingResult: VideoBallTrackingResultResponse | null,
  requestedKey: BallDetectorModelKey
): boolean {
  const detector = detectionResult?.summary.detector;
  if (!detector || !trackingResult) return false;
  return detector.requested_key === requestedKey;
}

function detectionPhase(status: VideoBallDetectionJobStatus): BallAnalysisPhase {
  if (status === "queued" || status === "loading_model") return "preparing_detector";
  if (status === "processing") return "detecting";
  return "selecting_candidates";
}

function trackingPhase(status: VideoBallTrackingJobStatus): BallAnalysisPhase {
  if (status === "queued" || status === "loading_detections") {
    return "filtering_static";
  }
  if (status === "analysing_candidates") return "selecting_candidates";
  if (status === "building_track") return "tracking";
  if (status === "recovering_gaps") return "recovering_gaps";
  if (status === "fitting_physics") return "scoring_primary";
  if (status === "rendering_video" || status === "saving_results") {
    return "preparing_review";
  }
  return "preparing_review";
}

function delay(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function nativeBox(box: NormalizedBox, width: number, height: number) {
  const xMin = Math.round(box.x * width);
  const yMin = Math.round(box.y * height);
  const xMax = Math.round((box.x + box.width) * width);
  const yMax = Math.round((box.y + box.height) * height);
  return {
    x_min: xMin,
    y_min: yMin,
    x_max: xMax,
    y_max: yMax,
    width: xMax - xMin,
    height: yMax - yMin,
    frame_width: width,
    frame_height: height
  };
}

function sourceLabel(source: BoxSource) {
  return source === "detected"
    ? "Detector"
    : source === "adjusted"
      ? "Detector adjusted"
      : "Manual";
}

function anchorSource(source: BoxSource): "DETECTOR" | "MANUAL" | "DETECTOR_ADJUSTED" {
  return source === "detected"
    ? "DETECTOR"
    : source === "adjusted"
      ? "DETECTOR_ADJUSTED"
      : "MANUAL";
}

export default function VideoAnalysisPage() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const ballRunGeneration = useRef(0);
  const playbackAnimationFrame = useRef<number | null>(null);
  const [analysis, setAnalysis] = useState<VideoAnalysisPreparedResponse | null>(null);
  const [detection, setDetection] = useState<VideoCalibrationDetectionResponse | null>(null);
  const [nearBox, setNearBox] = useState<NormalizedBox>(NEAR_DEFAULT);
  const [farBox, setFarBox] = useState<NormalizedBox>(FAR_DEFAULT);
  const [nearSource, setNearSource] = useState<BoxSource>("manual");
  const [farSource, setFarSource] = useState<BoxSource>("manual");
  const [savedCalibration, setSavedCalibration] = useState<ConfirmedVideoCalibrationResponse | null>(null);
  const [bridge, setBridge] = useState<NormalizedCameraBridgeResponse | null>(null);
  const [pitchModel, setPitchModel] = useState<VirtualPitchModel | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{ width: number; height: number } | null>(null);
  const [uploading, setUploading] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [fitting, setFitting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [editing, setEditing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ballPhase, setBallPhase] = useState<BallAnalysisPhase>("idle");
  const [ballProgress, setBallProgress] = useState(0);
  const [ballBusy, setBallBusy] = useState(false);
  const [ballError, setBallError] = useState<string | null>(null);
  const [ballTrack, setBallTrack] = useState<VideoBallTrackingResultResponse | null>(null);
  const [ballDetection, setBallDetection] = useState<VideoBallDetectionResultResponse | null>(null);
  const [ballDetectorKey, setBallDetectorKey] = useState<BallDetectorModelKey>("automatic");
  const [detectorModels, setDetectorModels] = useState<BallDetectorModelOption[]>([]);
  const [ballDetectionSummary, setBallDetectionSummary] = useState<VideoBallDetectionSummary | null>(null);
  const [displayToggles, setDisplayToggles] = useState<BallReviewDisplayToggles>(
    DEFAULT_BALL_REVIEW_TOGGLES
  );
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0);
  const [activeFrame, setActiveFrame] = useState(0);

  async function upload(file: File) {
    if (!file.type.startsWith("video/")) {
      setError("Choose a video file to begin scene calibration.");
      return;
    }
    setUploading(true);
    setError(null);
    setAnalysis(null);
    setDetection(null);
    setSavedCalibration(null);
    setBridge(null);
    setPitchModel(null);
    setVideoDimensions(null);
    setEditing(true);
    ballRunGeneration.current += 1;
    setBallPhase("idle");
    setBallProgress(0);
    setBallBusy(false);
    setBallError(null);
    setBallTrack(null);
    setBallDetection(null);
    setBallDetectionSummary(null);
    setDisplayToggles(DEFAULT_BALL_REVIEW_TOGGLES);
    setBallDetectorKey("automatic");
    setDetectorModels([]);
    setCurrentTimeSeconds(0);
    setActiveFrame(0);
    try {
      const prepared = await prepareVideoAnalysis(file);
      setAnalysis(prepared);
      setDetecting(true);
      const result = await detectVideoAnalysisCalibration(prepared.analysis_id, {
        frameZeroOnly: true,
        strikerGuide: FAR_DEFAULT,
        nonStrikerGuide: NEAR_DEFAULT
      });
      setDetection(result);
      setAnalysis((current) => current ? {
        ...current,
        reference_frame_index: result.reference_frame_index,
        reference_frame_url: result.reference_frame_url
      } : current);
      const detectedFar = result.provisional_striker_wicket;
      const detectedNear = result.provisional_non_striker_wicket;
      setFarBox(detectedFar?.box ?? FAR_DEFAULT);
      setNearBox(detectedNear?.box ?? NEAR_DEFAULT);
      setFarSource(detectedFar ? "detected" : "manual");
      setNearSource(detectedNear ? "detected" : "manual");
    } catch (caught) {
      setError(errorMessage(caught, "The video could not be prepared."));
    } finally {
      setUploading(false);
      setDetecting(false);
    }
  }

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void upload(file);
    event.target.value = "";
  }

  function dropVideo(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void upload(file);
  }

  function updateBox(role: "striker" | "non_striker", box: NormalizedBox) {
    if (role === "striker") {
      setFarBox(box);
      setFarSource((source) => source === "manual" ? "manual" : "adjusted");
    } else {
      setNearBox(box);
      setNearSource((source) => source === "manual" ? "manual" : "adjusted");
    }
  }

  function resetBoxes() {
    const detectedFar = detection?.provisional_striker_wicket;
    const detectedNear = detection?.provisional_non_striker_wicket;
    setFarBox(detectedFar?.box ?? FAR_DEFAULT);
    setNearBox(detectedNear?.box ?? NEAR_DEFAULT);
    setFarSource(detectedFar ? "detected" : "manual");
    setNearSource(detectedNear ? "detected" : "manual");
    setError(null);
  }

  async function pollDetection(
    analysisId: string,
    jobId: string,
    generation: number
  ): Promise<VideoBallDetectionResultResponse | null> {
    while (ballRunGeneration.current === generation) {
      const current = await getVideoBallDetectionJob(analysisId, jobId);
      if (ballRunGeneration.current !== generation) return null;
      setBallPhase(detectionPhase(current.status));
      setBallProgress(current.progress);
      if (current.status === "ready") {
        const result = await getVideoBallDetectionResult(analysisId);
        if (!result) throw new Error("BALL_DETECTION_FAILED: Completed detections could not be loaded.");
        return result;
      }
      if (current.status === "ball_detector_missing") {
        throw new Error(`${current.failure_code ?? "BALL_DETECTOR_UNAVAILABLE"}: ${current.error_message ?? current.message}`);
      }
      if (current.status === "failed") {
        throw new Error(`${current.failure_code ?? "BALL_DETECTION_FAILED"}: ${current.error_message ?? current.message}`);
      }
      await delay(POLL_INTERVAL_MS);
    }
    return null;
  }

  async function loadCompletedAnalysis(
    analysisId: string,
    includeDetectionFrames = true
  ) {
    const [detectionResult, trackingResult] = await Promise.all([
      getVideoBallDetectionResult(analysisId, includeDetectionFrames),
      getVideoBallTrackingResult(analysisId)
    ]);
    if (detectionResult) {
      setBallDetection(detectionResult);
      setBallDetectionSummary(detectionResult.summary);
      const requestedKey = detectionResult.summary.detector?.requested_key;
      if (
        requestedKey === "automatic"
        || requestedKey === "e2"
        || requestedKey === "e3"
        || requestedKey === "e4c"
      ) {
        setBallDetectorKey(requestedKey);
      }
    }
    if (trackingResult) {
      setBallTrack(trackingResult);
    }
    return { detectionResult, trackingResult };
  }

  async function pollTracking(
    analysisId: string,
    jobId: string,
    generation: number
  ): Promise<VideoBallTrackingResultResponse | null> {
    while (ballRunGeneration.current === generation) {
      const current = await getVideoBallTrackingJob(analysisId, jobId);
      if (ballRunGeneration.current !== generation) return null;
      setBallPhase(trackingPhase(current.status));
      setBallProgress(current.progress);
      if (current.status === "ready" || current.status === "no_reliable_track") {
        const result = await getVideoBallTrackingResult(analysisId);
        if (!result) throw new Error("TRACK_RESULT_LOAD_FAILED: Completed tracking could not be loaded.");
        return result;
      }
      if (current.status === "failed") {
        throw new Error(`${current.failure_code ?? "TRACK_UNAVAILABLE"}: ${current.error_message ?? current.message}`);
      }
      await delay(POLL_INTERVAL_MS);
    }
    return null;
  }

  async function runBallAnalysis(rerun = false) {
    if (!analysis || ballBusy) return;
    if (!Number.isFinite(analysis.fps) || analysis.fps <= 0) {
      setBallPhase("failed");
      setBallError("VIDEO_FPS_UNAVAILABLE: The source video has no usable frame rate.");
      return;
    }

    const generation = ballRunGeneration.current + 1;
    ballRunGeneration.current = generation;
    setBallBusy(true);
    setBallError(null);
    setBallProgress(0);
    setBallPhase("preparing_detector");

    try {
      if (!rerun) {
        try {
          const persisted = await loadCompletedAnalysis(analysis.analysis_id);
          if (
            persisted.trackingResult
            && persisted.detectionResult
            && persistedResultsMatch(
              persisted.detectionResult,
              persisted.trackingResult,
              ballDetectorKey
            )
          ) {
            setBallProgress(100);
            if (
              persisted.trackingResult.status === "ready"
              && persisted.trackingResult.primary_track.length > 0
            ) {
              setBallPhase("ready");
              return;
            }
            if (persisted.detectionResult.summary.total_candidates > 0) {
              setBallPhase("failed");
              setBallError(`TRACK_UNAVAILABLE: ${persisted.trackingResult.message}`);
              return;
            }
            setBallPhase("failed");
            setBallError(`TRACK_UNAVAILABLE: ${persisted.trackingResult.message}`);
            return;
          }
        } catch (caught) {
          throw new Error(`TRACK_RESULT_LOAD_FAILED: ${errorMessage(caught, "Saved tracking could not be loaded.")}`);
        }
      } else {
        setBallTrack(null);
        setBallDetection(null);
        setBallDetectionSummary(null);
      }

      const started = await startVideoBallDetection(analysis.analysis_id, ballDetectorKey);
      const detectionResult = await pollDetection(analysis.analysis_id, started.job_id, generation);
      if (!detectionResult || ballRunGeneration.current !== generation) return;

      const detailedDetection = await getVideoBallDetectionResult(
        analysis.analysis_id,
        true
      );
      if (detailedDetection) {
        setBallDetection(detailedDetection);
        setBallDetectionSummary(detailedDetection.summary);
      } else {
        setBallDetection(detectionResult);
        setBallDetectionSummary(detectionResult.summary);
      }

      if ((detailedDetection ?? detectionResult).summary.total_candidates <= 0) {
        throw new Error("NO_MOVING_BALL_CANDIDATES: Detection completed without a usable ball candidate.");
      }

      setBallPhase("tracking");
      const startedTracking = await startVideoBallTracking(analysis.analysis_id);
      const trackingResult = await pollTracking(
        analysis.analysis_id,
        startedTracking.job_id,
        generation
      );
      if (!trackingResult || ballRunGeneration.current !== generation) return;
      setBallTrack(trackingResult);
      setBallProgress(100);
      if (trackingResult.status !== "ready" || trackingResult.primary_track.length === 0) {
        setBallPhase("failed");
        setBallError(`TRACK_UNAVAILABLE: ${trackingResult.message}`);
        return;
      }
      setBallPhase("ready");
    } catch (caught) {
      if (ballRunGeneration.current !== generation) return;
      setBallPhase("failed");
      setBallError(errorMessage(caught, "TRACK_UNAVAILABLE: Ball tracking failed."));
    } finally {
      if (ballRunGeneration.current === generation) setBallBusy(false);
    }
  }

  function syncPlaybackPosition(video: HTMLVideoElement) {
    const nextTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
    setCurrentTimeSeconds(nextTime);
    setActiveFrame(Math.max(0, Math.floor(nextTime * (analysis?.fps ?? 0) + 1e-6)));
  }

  function stopPlaybackClock() {
    if (playbackAnimationFrame.current !== null) {
      window.cancelAnimationFrame(playbackAnimationFrame.current);
      playbackAnimationFrame.current = null;
    }
  }

  function startPlaybackClock(video: HTMLVideoElement) {
    stopPlaybackClock();
    const tick = () => {
      syncPlaybackPosition(video);
      if (!video.paused && !video.ended) {
        playbackAnimationFrame.current = window.requestAnimationFrame(tick);
      }
    };
    tick();
  }

  useEffect(() => () => {
    ballRunGeneration.current += 1;
    if (playbackAnimationFrame.current !== null) {
      window.cancelAnimationFrame(playbackAnimationFrame.current);
    }
  }, []);

  useEffect(() => {
    if (!analysis || !savedCalibration || !bridge || bridge.fitStatus === "FIT_FAILED" || editing) {
      return;
    }
    let cancelled = false;
    void getBallDetectorModels()
      .then((response) => {
        if (!cancelled) setDetectorModels(response.models);
      })
      .catch(() => {
        if (!cancelled) setDetectorModels([]);
      });
    return () => {
      cancelled = true;
    };
  }, [analysis, bridge, editing, savedCalibration]);

  useEffect(() => {
    const pitchReady = Boolean(
      savedCalibration
      && bridge
      && bridge.fitStatus !== "FIT_FAILED"
      && !editing
    );
    if (!pitchReady || !analysis || ballBusy || ballPhase !== "idle") return;
    let cancelled = false;
    void (async () => {
      try {
        const persisted = await loadCompletedAnalysis(analysis.analysis_id);
        if (cancelled) return;
        if (
          !persisted.detectionResult
          || !persisted.trackingResult
          || !persistedResultsMatch(
            persisted.detectionResult,
            persisted.trackingResult,
            ballDetectorKey
          )
        ) {
          return;
        }
        setBallProgress(100);
        if (
          persisted.trackingResult.status === "ready"
          && persisted.trackingResult.primary_track.length > 0
        ) {
          setBallPhase("ready");
        }
      } catch {
        // No saved ball analysis yet.
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis?.analysis_id, savedCalibration, bridge, editing, ballBusy, ballPhase, ballDetectorKey]);

  async function confirmAndFit() {
    if (!analysis) return;
    setFitting(true);
    setError(null);
    try {
      const nativeWidth = analysis.width;
      const nativeHeight = analysis.height;
      const nearPixels = nativeBox(nearBox, nativeWidth, nativeHeight);
      const farPixels = nativeBox(farBox, nativeWidth, nativeHeight);
      const nativeEvidence = {
        near_wicket: {
          ...nearPixels,
          role: "NEAR_WICKET" as const,
          source: anchorSource(nearSource),
          detector_confidence: detection?.provisional_non_striker_wicket?.confidence ?? null
        },
        far_wicket: {
          ...farPixels,
          role: "FAR_WICKET" as const,
          source: anchorSource(farSource),
          detector_confidence: detection?.provisional_striker_wicket?.confidence ?? null
        }
      };
      const confirmed = await confirmVideoAnalysisCalibration(analysis.analysis_id, {
        analysis_id: analysis.analysis_id,
        striker_wicket: {
          label: "striker",
          source: farSource,
          confidence: detection?.provisional_striker_wicket?.confidence ?? null,
          box: farBox,
          detection_pass: detection?.provisional_striker_wicket?.detection_pass ?? null
        },
        non_striker_wicket: {
          label: "non_striker",
          source: nearSource,
          confidence: detection?.provisional_non_striker_wicket?.confidence ?? null,
          box: nearBox,
          detection_pass: detection?.provisional_non_striker_wicket?.detection_pass ?? null
        },
        corridor_width_multiplier: 1,
        striker_guide: farBox,
        non_striker_guide: nearBox,
        render_scene_overlay: false,
        user_note: JSON.stringify({ coordinate_space: "FRAME_0_NATIVE_PIXELS", ...nativeEvidence })
      });
      setSavedCalibration(confirmed);

      const [specification, presets] = await Promise.all([
        getVirtualPitchSpecification(),
        getCameraSetupPresets()
      ]);
      setPitchModel(adaptVirtualPitchResponse(specification));
      const presetId = presets.find(
        (preset) => preset.preset_id === "STANDARD_REAR_WICKET_NET_V1"
      )?.preset_id;
      if (!presetId) {
        throw new Error("The standard rear-wicket camera preset is unavailable.");
      }
      const fittedBridge = await fitConfirmedWicketCamera(analysis.analysis_id, {
        preset_id: presetId,
        near_wicket: {
          x_min: nearPixels.x_min,
          y_min: nearPixels.y_min,
          x_max: nearPixels.x_max,
          y_max: nearPixels.y_max,
          width: nearPixels.width,
          height: nearPixels.height,
          frame_width: nearPixels.frame_width,
          frame_height: nearPixels.frame_height,
          role: nativeEvidence.near_wicket.role,
          source: nativeEvidence.near_wicket.source,
          detector_confidence: nativeEvidence.near_wicket.detector_confidence
        },
        far_wicket: {
          x_min: farPixels.x_min,
          y_min: farPixels.y_min,
          x_max: farPixels.x_max,
          y_max: farPixels.y_max,
          width: farPixels.width,
          height: farPixels.height,
          frame_width: farPixels.frame_width,
          frame_height: farPixels.frame_height,
          role: nativeEvidence.far_wicket.role,
          source: nativeEvidence.far_wicket.source,
          detector_confidence: nativeEvidence.far_wicket.detector_confidence
        }
      });
      setBridge(fittedBridge);
      if (fittedBridge.fitStatus === "FIT_FAILED") {
        setEditing(true);
        setError(fitFailureMessage(fittedBridge));
      } else {
        setEditing(false);
        setError(null);
      }
    } catch (caught) {
      setError(errorMessage(caught, "The pitch could not be fitted."));
    } finally {
      setFitting(false);
    }
  }

  const fitReady = Boolean(
    savedCalibration
    && bridge
    && bridge.fitStatus !== "FIT_FAILED"
    && !editing
  );
  const mediaWidth = videoDimensions?.width ?? analysis?.width ?? 16;
  const mediaHeight = videoDimensions?.height ?? analysis?.height ?? 9;
  const aspectRatio = mediaWidth / mediaHeight;
  const cameraPreset = useMemo(
    () => pitchModel ? calculateCameraPreset("setup", pitchModel, aspectRatio) : null,
    [aspectRatio, pitchModel]
  );
  const cameraDimensionsMatch = Boolean(
    bridge
    && bridge.camera.image_width === mediaWidth
    && bridge.camera.image_height === mediaHeight
  );
  const useRenderedOverlay = Boolean(
    bridge
    && pitchModel
    && cameraPreset
    && cameraDimensionsMatch
    && bridge.fitStatus !== "FIT_FAILED"
  );
  const trackPoints = useMemo(
    () => ballTrack?.primary_track ?? [],
    [ballTrack?.primary_track]
  );
  const reviewCandidates = useMemo(
    () => buildReviewCandidates(
      ballDetection?.frames,
      ballTrack?.candidate_diagnostics
    ),
    [ballDetection?.frames, ballTrack?.candidate_diagnostics]
  );
  const frameReviewCandidates = useMemo(
    () => candidatesForFrame(reviewCandidates, activeFrame),
    [reviewCandidates, activeFrame]
  );
  const framePrimaryPoint = useMemo(
    () => activeFramePrimaryPoint(trackPoints, activeFrame),
    [trackPoints, activeFrame]
  );
  const timelineMarkers = useMemo(
    () => buildTimelineMarkers(
      trackPoints,
      [...new Set(reviewCandidates.map((candidate) => candidate.frame_index))],
      [...new Set(reviewCandidates.filter((candidate) => candidate.selected).map((candidate) => candidate.frame_index))],
      [...new Set(reviewCandidates.filter((candidate) => !candidate.selected).map((candidate) => candidate.frame_index))]
    ),
    [reviewCandidates, trackPoints]
  );
  const observedPointCount = trackPoints.filter(
    (point) => point.provenance === "OBSERVED"
  ).length;
  const recoveredPointCount = trackPoints.length - observedPointCount;
  const fitStatusLabel = bridge?.fitStatus ? FIT_STATUS_LABELS[bridge.fitStatus] : null;
  const resolvedDetector = resolvedDetectorLabel(ballDetectionSummary, detectorModels);
  const selectedDetectorLabel = detectorModels.find((model) => model.key === ballDetectorKey)?.display_name
    ?? ballDetectionSummary?.detector?.display_name
    ?? ballDetectorKey;

  function seekToFrame(frameIndex: number) {
    const video = videoRef.current;
    if (!video || !analysis) return;
    const nextTime = Math.max(0, frameIndex / analysis.fps);
    video.currentTime = nextTime;
    syncPlaybackPosition(video);
  }

  return (
    <div className="mx-auto max-w-[90rem] py-2">
      <header className="border-b border-white/10 pb-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs font-bold uppercase text-lime">Video Analysis</p>
            <h1 className="mt-1 text-2xl font-black text-white">
              {analysis ? analysis.original_filename : "Fit the pitch to your video"}
            </h1>
            {analysis ? (
              <p className="mt-1 text-xs text-white/45">
                {analysis.width} x {analysis.height} · {analysis.fps.toFixed(2)} FPS · {analysis.analysis_id}
              </p>
            ) : null}
          </div>
          {fitReady ? (
            <div className="text-right text-xs text-white/55">
              <p>{fitStatusLabel ?? bridge?.fitStatus}</p>
              <p className="mt-1">
                Detector: {selectedDetectorLabel}
                {resolvedDetector ? ` · ${resolvedDetector}` : ""}
              </p>
              <p className="mt-1 capitalize">{ballPhase === "ready" ? "Analysis ready" : BALL_PROGRESS_LABELS[ballPhase] || "Ready for analysis"}</p>
            </div>
          ) : null}
        </div>
      </header>

      {!analysis ? (
        <div
          className={`mt-6 grid min-h-[24rem] place-items-center border-2 border-dashed p-8 text-center transition ${dragging ? "border-lime bg-lime/10" : "border-white/15 bg-white/[0.025]"}`}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={dropVideo}
        >
          <div className="max-w-md">
            <p className="text-xl font-bold text-white">Drop a cricket video here</p>
            <p className="mt-2 text-sm text-white/50">or choose a video from this computer</p>
            <Button className="mt-5" disabled={uploading} onClick={() => inputRef.current?.click()}>
              {uploading ? "Uploading..." : "Browse videos"}
            </Button>
            <input ref={inputRef} className="sr-only" type="file" accept="video/*" onChange={chooseFile} />
          </div>
        </div>
      ) : (
        <div className="mt-6 space-y-6">
          {!fitReady ? (
            <section>
              <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h2 className="text-xl font-bold text-white">Scene Calibration</h2>
                  <p className="mt-1 text-sm text-white/50">Frame {analysis.reference_frame_index}. Position each box around the complete wicket.</p>
                </div>
                <Button variant="secondary" onClick={resetBoxes}>Reset boxes</Button>
              </div>

              <CalibrationCanvas
                imageUrl={analysis.reference_frame_url}
                imageWidth={analysis.width}
                imageHeight={analysis.height}
                striker={null}
                nonStriker={null}
                strikerGuide={farBox}
                nonStrikerGuide={nearBox}
                pitchGeometry={null}
                interactionMode="guides"
                showGuides
                guideLabels={{ striker: "Far Wicket", non_striker: "Near Wicket" }}
                onGuideChange={updateBox}
              />

              <div className="mt-4 grid gap-3 border-y border-white/10 py-4 sm:grid-cols-2">
                <WicketReadout title="Near Wicket" source={nearSource} box={nearBox} width={mediaWidth} height={mediaHeight} />
                <WicketReadout title="Far Wicket" source={farSource} box={farBox} width={mediaWidth} height={mediaHeight} />
              </div>
              <div className="mt-5 flex flex-wrap items-center gap-3">
                <Button disabled={detecting || fitting} onClick={() => void confirmAndFit()}>
                  {detecting
                    ? "Detecting wickets..."
                    : fitting
                      ? "Placing pitch..."
                      : bridge
                        ? "Refit Pitch"
                        : "Confirm Stumps and Fit Pitch"}
                </Button>
                <span className="text-sm text-white/45">
                  {detection?.status === "candidates_ready" ? "Detector suggestions ready" : "Manual placement available"}
                </span>
              </div>
            </section>
          ) : (
            <section
              data-camera-source={bridge?.camera.source_version}
              data-camera-fit-status={bridge?.fitStatus}
            >
              <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" onClick={() => setEditing(true)}>Adjust Boxes</Button>
                  <Button variant="secondary" disabled={fitting} onClick={() => void confirmAndFit()}>
                    {fitting ? "Refitting..." : "Refit Pitch"}
                  </Button>
                  <Button variant="secondary" onClick={() => inputRef.current?.click()}>Choose another video</Button>
                  <input ref={inputRef} className="sr-only" type="file" accept="video/*" onChange={chooseFile} />
                </div>
                {bridge?.fitStatus === "FIT_APPROXIMATE" ? (
                  <p className="text-xs text-amber-200/90">Approximate pitch fit — review overlay if needed.</p>
                ) : null}
              </div>

              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(17rem,22rem)]">
                <div className="min-w-0">
                  <div className="relative w-full overflow-hidden bg-black" style={{ aspectRatio: `${mediaWidth} / ${mediaHeight}` }}>
                    <video
                      ref={videoRef}
                      className="absolute inset-0 h-full w-full object-contain"
                      src={analysis.playback_video_url ?? analysis.original_video_url}
                      controls
                      playsInline
                      onLoadedMetadata={(event) => {
                        const video = event.currentTarget;
                        setVideoDimensions({ width: video.videoWidth, height: video.videoHeight });
                        syncPlaybackPosition(video);
                      }}
                      onPlay={(event) => startPlaybackClock(event.currentTarget)}
                      onPause={(event) => {
                        stopPlaybackClock();
                        syncPlaybackPosition(event.currentTarget);
                      }}
                      onEnded={(event) => {
                        stopPlaybackClock();
                        syncPlaybackPosition(event.currentTarget);
                      }}
                      onTimeUpdate={(event) => syncPlaybackPosition(event.currentTarget)}
                      onSeeking={(event) => syncPlaybackPosition(event.currentTarget)}
                      onSeeked={(event) => syncPlaybackPosition(event.currentTarget)}
                    />
                    {useRenderedOverlay && pitchModel && cameraPreset && bridge ? (
                      <div className="pointer-events-none absolute inset-0">
                        <VirtualPitchCanvas
                          model={pitchModel}
                          mode="real-frame-overlay"
                          camera={cameraPreset}
                          calibratedCamera={bridge.camera}
                          visualOptions={{
                            showPitch: true,
                            showStumps: true,
                            showBails: true,
                            showLines: true,
                            showCorridor: true,
                            showAxes: false,
                            showGrid: false,
                            enableOrbitControls: false,
                            corridorOpacity: 0.24,
                            lowPerformance: false,
                            dprCap: 2,
                            overlayOpacity: 0.76,
                            materialPreset: materialPreset("cricvision-dark")
                          }}
                        />
                      </div>
                    ) : null}
                    {(trackPoints.length > 0 || reviewCandidates.length > 0) ? (
                      <BallTrackOverlay
                        points={trackPoints}
                        candidates={reviewCandidates}
                        toggles={displayToggles}
                        currentTimeSeconds={currentTimeSeconds}
                        currentFrame={activeFrame}
                        nativeWidth={mediaWidth}
                        nativeHeight={mediaHeight}
                      />
                    ) : null}
                  </div>

                  <BallAnalysisTimeline
                    totalFrames={analysis.frame_count ?? ballDetectionSummary?.total_frames ?? 0}
                    fps={analysis.fps}
                    currentFrame={activeFrame}
                    currentTimeSeconds={currentTimeSeconds}
                    markers={timelineMarkers}
                    onSeekFrame={seekToFrame}
                  />
                </div>

                <BallAnalysisReviewPanel
                  ballDetectorKey={ballDetectorKey}
                  detectorModels={detectorModels}
                  detectionSummary={ballDetectionSummary}
                  trackingResult={ballTrack}
                  reviewCandidates={reviewCandidates}
                  activeFrameCandidates={frameReviewCandidates}
                  activePrimaryPoint={framePrimaryPoint}
                  ballPhase={ballPhase}
                  ballProgress={ballProgress}
                  ballBusy={ballBusy}
                  ballError={ballError}
                  toggles={displayToggles}
                  onToggleChange={(key, value) => {
                    setDisplayToggles((current) => ({ ...current, [key]: value }));
                  }}
                  onDetectorChange={(key) => {
                    setBallDetectorKey(key);
                    if (ballDetectionSummary?.detector?.requested_key !== key) {
                      setBallTrack(null);
                      setBallDetection(null);
                      setBallDetectionSummary(null);
                      setBallPhase("idle");
                      setBallProgress(0);
                      setBallError(null);
                    }
                  }}
                  pipelineVersion="delivery_track_v2"
                  onRun={(rerun) => void runBallAnalysis(rerun)}
                  selectedDetectorLabel={selectedDetectorLabel}
                  resolvedDetector={resolvedDetector}
                />
              </div>

              {trackPoints.length > 0 ? (
                <p className="mt-3 text-xs text-white/40">
                  {trackPoints.length} track points · {observedPointCount} observed · {recoveredPointCount} recovered/projected
                </p>
              ) : null}
            </section>
          )}
        </div>
      )}

      {error ? <p role="alert" className="mt-4 border border-signal/40 bg-signal/10 px-4 py-3 text-sm text-[#ffaaa6]">{error}</p> : null}
    </div>
  );
}

function WicketReadout({
  title,
  source,
  box,
  width,
  height
}: {
  title: string;
  source: BoxSource;
  box: NormalizedBox;
  width: number;
  height: number;
}) {
  const pixels = nativeBox(box, width, height);
  return (
    <div className="flex items-center justify-between gap-4 px-1 text-sm">
      <div>
        <p className="font-bold text-white">{title}</p>
        <p className="mt-1 text-xs text-white/40">{sourceLabel(source)}</p>
      </div>
      <p className="text-right font-mono text-xs text-white/65">
        {pixels.x_min}, {pixels.y_min} to {pixels.x_max}, {pixels.y_max}
      </p>
    </div>
  );
}
