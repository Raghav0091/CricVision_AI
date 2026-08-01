import { Matrix4, PerspectiveCamera } from "three";

import {
  cameraFamilyForBridge,
  configureCalibratedCamera,
  matrixChecksum,
  shouldMountOrbitControls,
  validateActiveRendererCamera,
  validRenderBounds,
  type OwnedPerspectiveCamera
} from "./cameraOwnership";
import { buildThreeCameraFromOpenCv, type CameraBridgeInput } from "./opencvCameraBridge";


function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}


const input: CameraBridgeInput = {
  source: "camera-ownership-test",
  source_version: "v1",
  accepted: true,
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
  far: 100
};
const bridge = buildThreeCameraFromOpenCv(input);
const camera = configureCalibratedCamera(
  new PerspectiveCamera() as OwnedPerspectiveCamera,
  bridge
);
const cameraUuid = camera.uuid;
const projectionChecksum = matrixChecksum(camera.projectionMatrix.elements);
const poseChecksum = matrixChecksum(camera.matrixWorld.elements);

assert(camera.manual === true, "Calibrated camera must opt out of R3F projection updates.");
assert(camera.matrixAutoUpdate === false, "Calibrated camera pose must be bridge-owned.");
assert(cameraFamilyForBridge(bridge) === "CALIBRATED_OPENCV_CAMERA", "Calibrated family selection failed.");
assert(cameraFamilyForBridge(null) === "DEVELOPMENT_CAMERA", "Development family selection failed.");
assert(!shouldMountOrbitControls("CALIBRATED_OPENCV_CAMERA", true), "Calibrated camera mounted orbit controls.");
assert(shouldMountOrbitControls("DEVELOPMENT_CAMERA", true), "Free-orbit controls were not mounted.");

for (const [width, height, dpr] of [[390, 844, 1], [1440, 900, 2], [844, 390, 1.5]]) {
  assert(validRenderBounds(width, height), "Valid responsive bounds were rejected.");
  // R3F's resize path skips cameras marked manual; CSS size and DPR do not mutate these matrices.
  void dpr;
  assert(camera.uuid === cameraUuid, "Camera identity changed during resize.");
  assert(matrixChecksum(camera.projectionMatrix.elements) === projectionChecksum, "Projection changed during resize.");
  assert(matrixChecksum(camera.matrixWorld.elements) === poseChecksum, "Pose changed during resize.");
}
assert(!validRenderBounds(0, 720), "Zero-width stage was accepted.");
assert(!validRenderBounds(1280, 0), "Zero-height stage was accepted.");

const report = validateActiveRendererCamera(camera, bridge, [
  { semanticId: "origin", world: { x: 0, y: 0, z: 0 } },
  { semanticId: "offset", world: { x: 1, y: 0, z: 1 } }
]);
assert((report.rmse ?? Infinity) < 1e-9, `Active renderer RMSE is too high: ${report.rmse}`);
assert(report.points.every((point) => (point.openCvToBridgeError ?? Infinity) < 1e-9), "OpenCV and bridge diverged.");
assert(report.points.every((point) => (point.bridgeToActiveCameraError ?? Infinity) < 1e-9), "Bridge and active camera diverged.");

const developmentCamera = new PerspectiveCamera(45, 1, 0.1, 100);
const before = matrixChecksum(developmentCamera.projectionMatrix.elements);
developmentCamera.aspect = 16 / 9;
developmentCamera.updateProjectionMatrix();
assert(matrixChecksum(developmentCamera.projectionMatrix.elements) !== before, "Development projection did not respond to aspect change.");

const altered = camera.clone() as OwnedPerspectiveCamera;
altered.projectionMatrix.copy(new Matrix4().identity());
const alteredReport = validateActiveRendererCamera(altered, bridge, [{ semanticId: "offset", world: { x: 1, y: 0, z: 1 } }]);
assert((alteredReport.rmse ?? 0) > 1, "Mismatched active camera was not detected.");
