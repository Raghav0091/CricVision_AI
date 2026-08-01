import { aspectAdjustedVerticalFov, calculateCameraPreset } from "./cameraPresets";
import { toCricketVector } from "./coordinates";
import { adaptVirtualPitchResponse } from "./geometryAdapter";
import { validVirtualPitchResponse } from "./testFixture.test-support";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const model = adaptVirtualPitchResponse(validVirtualPitchResponse());
const landscapeSetup = calculateCameraPreset("setup", model, 16 / 9);
const portraitSetup = calculateCameraPreset("setup", model, 9 / 16);
const setupPosition = toCricketVector(landscapeSetup.position);
const setupTarget = toCricketVector(landscapeSetup.target);
assert(setupPosition.y < 0, "Setup camera must be behind the bowler-end wicket.");
assert(setupPosition.z >= 1.2 && setupPosition.z <= 1.3, "Setup camera height left the development preset range.");
assert(setupTarget.y > setupPosition.y, "Setup camera must look toward the striker end.");
assert(portraitSetup.verticalFovDegrees > landscapeSetup.verticalFovDegrees, "Portrait framing must widen vertical FOV.");
assert(aspectAdjustedVerticalFov(48, 9 / 16) > aspectAdjustedVerticalFov(48, 16 / 9), "Aspect adjustment is reversed.");

const bowler = calculateCameraPreset("bowler-end", model, 16 / 9);
const striker = calculateCameraPreset("striker-end", model, 16 / 9);
const bowlerPosition = toCricketVector(bowler.position);
const bowlerTarget = toCricketVector(bowler.target);
const strikerPosition = toCricketVector(striker.position);
const strikerTarget = toCricketVector(striker.target);
assert(bowlerPosition.y < bowlerTarget.y, "Bowler camera direction is reversed.");
assert(strikerPosition.y > strikerTarget.y, "Striker camera direction is reversed.");

const topDown = calculateCameraPreset("top-down", model, 390 / 844);
const topPosition = toCricketVector(topDown.position);
assert(topPosition.z > model.dimensions.stumpHeightM, "Top-down camera is not above the pitch.");
assert(topDown.up.z === -1, "Top-down camera must keep the striker direction screen-up.");
assert(Number.isFinite(topDown.far) && topDown.far > topDown.near, "Camera clipping planes are invalid.");

const free = calculateCameraPreset("free-orbit", model, 1);
assert(free.orbitEnabled, "Free Orbit must enable orbit controls.");
assert(!landscapeSetup.orbitEnabled, "Setup camera must remain deterministic.");
