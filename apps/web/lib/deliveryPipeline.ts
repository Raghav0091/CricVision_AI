import {
  acceptWicketBoxCalibration,
  detectWicketBoxCalibration,
  getReplayPayload,
  getVideoBallDetectionJob,
  getVideoBallTrackingJob,
  prepareVideoAnalysis,
  registerWicketBoxCalibration,
  startVideoBallDetection,
  startVideoBallTracking,
  type ReplayPayloadV1,
  type VideoAnalysisPreparedResponse
} from "@/lib/api";
import { getDeviceId } from "@/lib/deviceIdentity";
import type { BoxLayout, KeypointLayout, StumpKeypointSet } from "@/lib/types";
import type {
  CricketPitchGeometry,
  StumpLandmark,
  WicketBox,
  WicketBoxCalibrationRegisterRequest
} from "@/lib/wicketCalibration/types";


// One live delivery walks the same chain the /video-analysis workspace does,
// because that is the only path that produces a metric camera pose and
// therefore the only path that can produce real speed/length numbers.
// /calibration/solve (used by the live alignment screen) returns bbox geometry
// with no pose, which is fine for drawing an overlay and useless for metrics.
export type DeliveryStage =
  | "uploading"
  | "calibrating"
  | "detecting_ball"
  | "tracking_ball"
  | "building_replay"
  | "ready"
  | "failed";

export type DeliveryProgress = {
  stage: DeliveryStage;
  /** 0-100 within the current stage, when the backend reports it. */
  percent?: number;
  message: string;
};

export type DeliveryAnalysis = {
  analysisId: string;
  prepared: VideoAnalysisPreparedResponse;
  replay: ReplayPayloadV1;
};

const POLL_INTERVAL_MS = 1200;
const POLL_TIMEOUT_MS = 5 * 60 * 1000;

const DETECTION_TERMINAL = new Set(["ready", "failed", "ball_detector_missing"]);
const TRACKING_TERMINAL = new Set(["ready", "failed", "no_reliable_track"]);


function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}


// The live screen holds boxes normalised against the preview; the calibration
// contract wants pixels in the analysis frame. Round outward so a stump sitting
// exactly on the border stays inside the box.
function toWicketBox(
  role: "NEAR" | "FAR",
  box: BoxLayout["striker"],
  width: number,
  height: number,
  frameIndex: number
): WicketBox {
  return {
    role,
    x: Math.floor(box.x * width),
    y: Math.floor(box.y * height),
    width: Math.ceil(box.width * width),
    height: Math.ceil(box.height * height),
    source_image_width: width,
    source_image_height: height,
    calibration_frame_index: frameIndex,
    validation_status: "PENDING"
  };
}


// The operator has already placed six points per wicket on the live screen.
// Re-running the stump detector over the recorded clip throws that away and
// asks the model to rediscover what a human already marked — and when the
// operator calibrated manually precisely because detection was struggling, it
// finds nothing and the register call fails with zero landmarks.
function landmarksFromKeypoints(
  keypoints: KeypointLayout,
  width: number,
  height: number
): StumpLandmark[] {
  const identities = [
    ["LEFT", "left_top", "left_base"],
    ["MIDDLE", "middle_top", "middle_base"],
    ["RIGHT", "right_top", "right_base"]
  ] as const;

  const forEnd = (set: StumpKeypointSet, role: "NEAR" | "FAR"): StumpLandmark[] =>
    identities.map(([identity, topKey, baseKey]) => {
      const top = { x: set[topKey].x * width, y: set[topKey].y * height };
      const base = { x: set[baseKey].x * width, y: set[baseKey].y * height };
      return {
        wicket_role: role,
        stump_identity: identity,
        base,
        top,
        centre: { x: (top.x + base.x) / 2, y: (top.y + base.y) / 2 },
        visibility: "VISIBLE",
        confidence: 1,
        provenance: "USER_CORRECTED"
      };
    });

  // NEAR is the bowler end the camera sits behind, which the live screen calls
  // non_striker. Getting this pair the wrong way round inverts the pitch.
  return [
    ...forEnd(keypoints.non_striker, "NEAR"),
    ...forEnd(keypoints.striker, "FAR")
  ];
}


async function pollJob<T extends { status: string; progress: number; message: string; error_message?: string | null }>(
  fetchJob: () => Promise<T>,
  terminal: Set<string>,
  stage: DeliveryStage,
  onProgress: (progress: DeliveryProgress) => void
): Promise<T> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  for (;;) {
    const job = await fetchJob();
    onProgress({ stage, percent: job.progress, message: job.message });
    if (terminal.has(job.status)) return job;
    if (Date.now() > deadline) {
      throw new Error(`${stage} timed out after ${Math.round(POLL_TIMEOUT_MS / 1000)}s.`);
    }
    await sleep(POLL_INTERVAL_MS);
  }
}


/**
 * Upload one recorded delivery and run it all the way to a replay payload.
 *
 * `boxLayout` is the same NEAR/FAR layout the operator already aligned on the
 * live screen — reusing it means the bowler calibrates once, not twice.
 *
 * `pitchGeometry` is the pitch the operator declared. Null means regulation,
 * which is what every full-size net sends.
 */
export async function runDeliveryAnalysis(
  clip: Blob,
  boxLayout: BoxLayout,
  onProgress: (progress: DeliveryProgress) => void = () => {},
  placedKeypoints?: KeypointLayout | null,
  pitchGeometry?: CricketPitchGeometry | null
): Promise<DeliveryAnalysis> {
  const extension = clip.type.includes("mp4") ? "mp4" : "webm";
  const file = new File([clip], `delivery.${extension}`, { type: clip.type || `video/${extension}` });

  onProgress({ stage: "uploading", message: "Uploading delivery…" });
  const prepared = await prepareVideoAnalysis(file);
  const { analysis_id: analysisId, width, height, reference_frame_index: frameIndex } = prepared;

  onProgress({ stage: "calibrating", message: "Registering camera pose…" });
  const request: WicketBoxCalibrationRegisterRequest = {
    analysis_id: analysisId,
    calibration_frame_index: frameIndex,
    source_image_width: width,
    source_image_height: height,
    near_wicket_box: toWicketBox("NEAR", boxLayout.non_striker, width, height, frameIndex),
    far_wicket_box: toWicketBox("FAR", boxLayout.striker, width, height, frameIndex),
    // Harmless when this phone has no profile: the solver falls back to its
    // focal sweep exactly as before.
    device_id: getDeviceId(),
    // Null means regulation, so the solver behaves exactly as it always has.
    pitch_geometry: pitchGeometry ?? null
  };

  // Prefer the points the operator placed. Only fall back to detection when the
  // live screen has none to hand.
  let landmarks: StumpLandmark[];
  if (placedKeypoints) {
    landmarks = landmarksFromKeypoints(placedKeypoints, width, height);
  } else {
    const detected = await detectWicketBoxCalibration(analysisId, request);
    landmarks = detected.stump_landmarks;
  }
  if (landmarks.length < 6) {
    throw new Error(
      "No stump points were available for this delivery. Re-run calibration on " +
        "the live screen before recording."
    );
  }

  const registered = await registerWicketBoxCalibration(analysisId, {
    ...request,
    stump_landmarks: landmarks
  });
  if (!registered.success) {
    throw new Error(registered.message || "Could not register the camera pose for this delivery.");
  }
  await acceptWicketBoxCalibration(analysisId, {
    analysis_id: analysisId,
    accept_registered_calibration: true
  });

  onProgress({ stage: "detecting_ball", message: "Detecting the ball…" });
  const detection = await startVideoBallDetection(analysisId);
  const detectionJob = await pollJob(
    () => getVideoBallDetectionJob(analysisId, detection.job_id),
    DETECTION_TERMINAL,
    "detecting_ball",
    onProgress
  );
  if (detectionJob.status !== "ready") {
    throw new Error(detectionJob.error_message || detectionJob.message || "Ball detection failed.");
  }

  onProgress({ stage: "tracking_ball", message: "Tracking the delivery…" });
  const tracking = await startVideoBallTracking(analysisId);
  const trackingJob = await pollJob(
    () => getVideoBallTrackingJob(analysisId, tracking.job_id),
    TRACKING_TERMINAL,
    "tracking_ball",
    onProgress
  );
  if (trackingJob.status !== "ready") {
    throw new Error(trackingJob.error_message || trackingJob.message || "No reliable track for this delivery.");
  }

  onProgress({ stage: "building_replay", message: "Building the replay…" });
  const replay = await getReplayPayload(analysisId);

  onProgress({ stage: "ready", percent: 100, message: "Delivery ready." });
  return { analysisId, prepared, replay };
}
