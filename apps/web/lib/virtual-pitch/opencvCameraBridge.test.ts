import { Matrix4, Vector3, Vector4 } from "three";
import {
  buildCalibratedProjectionMatrix,
  buildProjectionMatrixInverse,
  calculateContainMapping,
  displayPixelToNative,
  nativePixelToDisplay,
  nativePixelToNdc,
  ndcToNativePixel
} from "./cameraProjection";
import { projectOpenCvWorldPoint, validateCameraBridge } from "./cameraValidation";
import {
  assessDistortion,
  buildThreeCameraFromOpenCv,
  type CameraBridgeInput
} from "./opencvCameraBridge";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function close(actual: number, expected: number, tolerance = 1e-9, label = "value") {
  assert(Math.abs(actual - expected) <= tolerance, `${label}: expected ${expected}, received ${actual}`);
}

function matrixIdentityError(matrix: Matrix4): number {
  const identity = new Matrix4().identity().elements;
  return Math.max(...matrix.elements.map((value, index) => Math.abs(value - identity[index])));
}

function camera(overrides: Partial<CameraBridgeInput> = {}): CameraBridgeInput {
  return {
    source: "synthetic_test",
    source_version: "v1",
    analysis_id: null,
    candidate_id: "exact",
    accepted: false,
    classification: "SYNTHETIC",
    image_width: 1280,
    image_height: 720,
    camera_matrix: [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
    distortion_coefficients: [0, 0, 0, 0, 0],
    rotation_representation: "matrix",
    rotation_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    translation_vector: [0, 0, 10],
    extrinsic_convention: "opencv_world_to_camera",
    world_coordinate_system: "cricvision_pitch_v1",
    near: 0.1,
    far: 100,
    ...overrides
  };
}

const centred = buildThreeCameraFromOpenCv(camera());
assert(centred.renderable && centred.exact, "Zero-distortion exact camera must be renderable.");
close(centred.cameraWorldPosition.x, 0, 1e-12, "identity camera x");
close(centred.cameraWorldPosition.y, -10, 1e-12, "identity camera Three y");
close(centred.cameraWorldPosition.z, 0, 1e-12, "identity camera Three z");
assert(centred.diagnostics.handednessPreserved, "Axis conversion changed handedness.");
assert(matrixIdentityError(centred.projectionMatrix.clone().multiply(centred.projectionMatrixInverse)) < 1e-10, "Projection inverse is inaccurate.");
assert(matrixIdentityError(centred.matrixWorld.clone().multiply(centred.matrixWorldInverse)) < 1e-10, "Extrinsic inverse is inaccurate.");

const landmarks = [
  { semanticId: "bowler-left-base", world: { x: -0.1143, y: 0, z: 0 } },
  { semanticId: "bowler-right-top", world: { x: 0.1143, y: 0, z: 0.711 } },
  { semanticId: "striker-left-base", world: { x: -0.1143, y: 20.12, z: 0 } },
  { semanticId: "striker-right-top", world: { x: 0.1143, y: 20.12, z: 0.711 } },
  { semanticId: "pitch-left-bowler", world: { x: -1.525, y: 0, z: 0 } },
  { semanticId: "pitch-right-striker", world: { x: 1.525, y: 20.12, z: 0 } },
  { semanticId: "bowling-crease-left", world: { x: -1.32, y: 0, z: 0 } },
  { semanticId: "popping-crease-right", world: { x: 1.32, y: 1.22, z: 0 } },
  { semanticId: "return-crease", world: { x: -1.32, y: 2.44, z: 0 } },
  { semanticId: "centreline-striker", world: { x: 0, y: 20.12, z: 0 } },
  { semanticId: "corridor-corner", world: { x: 0.3, y: 18, z: 0 } }
];
const exactReport = validateCameraBridge(centred, landmarks);
assert(exactReport.metrics.validPointCount === landmarks.length, "Exact test unexpectedly excluded visible points.");
assert((exactReport.metrics.rmse ?? Infinity) < 1e-9, `Exact RMSE too high: ${exactReport.metrics.rmse}`);
assert((exactReport.metrics.maximumError ?? Infinity) < 1e-8, `Exact maximum error too high: ${exactReport.metrics.maximumError}`);
assert(!exactReport.metrics.mirroredAxisWarning, "Exact camera was falsely marked mirrored.");
assert(!exactReport.metrics.bowlerStrikerReversalWarning, "Exact camera reversed pitch ends.");

const variedCameras: CameraBridgeInput[] = [
  camera({ camera_matrix: [[1100, 17, 594], [0, 730, 301], [0, 0, 1]] }),
  camera({ image_width: 720, image_height: 1280, camera_matrix: [[680, 0, 309], [0, 940, 671], [0, 0, 1]] }),
  camera({ rotation_matrix: [[0.939692621, 0, 0.342020143], [0, 1, 0], [-0.342020143, 0, 0.939692621]], translation_vector: [1.2, -0.4, 14] }),
  camera({ rotation_matrix: [[1, 0, 0], [0, 0.984807753, -0.173648178], [0, 0.173648178, 0.984807753]], translation_vector: [-1, 0.7, 18] })
];
for (const [index, input] of variedCameras.entries()) {
  const report = validateCameraBridge(buildThreeCameraFromOpenCv(input), landmarks);
  assert((report.metrics.rmse ?? Infinity) < 1e-7, `Varied camera ${index} RMSE too high: ${report.metrics.rmse}`);
  assert((report.metrics.maximumError ?? Infinity) < 1e-6, `Varied camera ${index} maximum error too high.`);
}

const translated = buildThreeCameraFromOpenCv(camera({ translation_vector: [2, -3, 15] }));
const roundTripPoint = new Vector4(2, 3, -4, 1)
  .applyMatrix4(translated.matrixWorldInverse)
  .applyMatrix4(translated.matrixWorld);
close(roundTripPoint.x, 2, 1e-10, "world round trip x");
close(roundTripPoint.y, 3, 1e-10, "world round trip y");
close(roundTripPoint.z, -4, 1e-10, "world round trip z");

// Camera at CricVision (0, -5, 2), looking from behind the bowler end toward +y.
const pitchFacing = buildThreeCameraFromOpenCv(camera({
  rotation_matrix: [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
  translation_vector: [0, 2, 5]
}));
close(pitchFacing.cameraWorldPosition.x, 0, 1e-10, "pitch-facing camera x");
close(pitchFacing.cameraWorldPosition.y, 2, 1e-10, "pitch-facing camera height");
close(pitchFacing.cameraWorldPosition.z, 5, 1e-10, "pitch-facing camera behind distance");
close(pitchFacing.cameraForwardDirection.x, 0, 1e-10, "pitch-facing forward x");
close(pitchFacing.cameraForwardDirection.y, 0, 1e-10, "pitch-facing forward y");
close(pitchFacing.cameraForwardDirection.z, -1, 1e-10, "pitch-facing forward z");
const pitchFacingReport = validateCameraBridge(pitchFacing, landmarks);
assert(pitchFacingReport.metrics.validPointCount === landmarks.length, "Pitch-facing camera lost pitch landmarks.");
assert((pitchFacingReport.metrics.rmse ?? Infinity) < 1e-9, "Pitch-facing camera projection diverged.");

const behindInput = camera({ translation_vector: [0, 0, -1] });
const behind = projectOpenCvWorldPoint(behindInput, { x: 0, y: 0, z: 0 });
assert(!behind.positiveDepth && behind.pixel === null, "Point behind the camera was projected as visible.");
const behindReport = validateCameraBridge(buildThreeCameraFromOpenCv(behindInput), [{ semanticId: "behind", world: { x: 0, y: 0, z: 0 } }]);
assert(behindReport.metrics.pointsBehindCamera === 1 && behindReport.metrics.validPointCount === 0, "Behind-camera accounting is incorrect.");

const centredProjection = buildCalibratedProjectionMatrix(centred.intrinsics, 0.1, 100);
const opticalAxis = new Vector4(0, 0, -10, 1).applyMatrix4(centredProjection);
close(opticalAxis.x / opticalAxis.w, 0, 1e-12, "centred principal point x NDC");
close(opticalAxis.y / opticalAxis.w, 0, 1e-12, "centred principal point y NDC");
assert(matrixIdentityError(centredProjection.clone().multiply(buildProjectionMatrixInverse(centredProjection))) < 1e-10, "Standalone projection inverse failed.");

const pixel = { x: 123.5, y: 456.25 };
const ndc = nativePixelToNdc(pixel, 1280, 720);
const pixelRoundTrip = ndcToNativePixel(ndc, 1280, 720);
close(pixelRoundTrip.x, pixel.x, 1e-12, "pixel/NDC x");
close(pixelRoundTrip.y, pixel.y, 1e-12, "pixel/NDC y");

for (const mapping of [
  calculateContainMapping(1920, 1080, 390, 844),
  calculateContainMapping(1080, 1920, 1440, 900),
  calculateContainMapping(1920, 1080, 1440, 900),
  calculateContainMapping(1080, 1920, 390, 844)
]) {
  const display = nativePixelToDisplay({ x: mapping.nativeWidth * 0.37, y: mapping.nativeHeight * 0.61 }, mapping);
  const native = displayPixelToNative(display, mapping);
  close(native.x, mapping.nativeWidth * 0.37, 1e-10, "contain x round trip");
  close(native.y, mapping.nativeHeight * 0.61, 1e-10, "contain y round trip");
}

assert(assessDistortion([0, 0, 0, 0, 0]).mode === "ZERO_DISTORTION", "Zero distortion policy failed.");
const unsupported = buildThreeCameraFromOpenCv(camera({ distortion_coefficients: [0.1, 0, 0, 0, 0] }));
assert(!unsupported.renderable && !unsupported.exact, "Non-zero raw distortion was silently accepted.");
assert(unsupported.diagnostics.distortion.mode === "NONZERO_DISTORTION_UNSUPPORTED", "Unsupported distortion mode missing.");
const undistorted = buildThreeCameraFromOpenCv(camera({ distortion_coefficients: [0.1, -0.03, 0, 0, 0], frame_preundistorted: true }));
assert(undistorted.renderable && undistorted.exact, "Pre-undistorted frame should be supported.");
assert(undistorted.diagnostics.distortion.mode === "PREUNDISTORTED_FRAME", "Pre-undistorted mode missing.");

let invalidClippingRejected = false;
try {
  buildCalibratedProjectionMatrix(centred.intrinsics, 10, 1);
} catch {
  invalidClippingRejected = true;
}
assert(invalidClippingRejected, "Invalid near/far planes were accepted.");

const forward = centred.cameraForwardDirection;
assert(new Vector3(forward.x, forward.y, forward.z).length() > 0.999999, "Camera forward direction is not normalized.");
