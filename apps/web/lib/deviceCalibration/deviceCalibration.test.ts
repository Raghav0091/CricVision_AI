// @ts-expect-error Node's type-stripping test runner requires an explicit TypeScript extension.
import { describeQuality } from "./quality.ts";
// @ts-expect-error Node's type-stripping test runner requires an explicit TypeScript extension.
import { getDeviceId, getDeviceLabel, setDeviceLabel } from "../deviceIdentity.ts";
import type { CalibrationQuality } from "./types";


function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}


// deviceIdentity reaches for window.localStorage, which Node does not have.
// A map behind the same three methods is enough to exercise persistence.
function installLocalStorage() {
  const entries = new Map<string, string>();
  (globalThis as { window?: unknown }).window = {
    localStorage: {
      getItem: (key: string) => entries.get(key) ?? null,
      setItem: (key: string, value: string) => void entries.set(key, value),
      removeItem: (key: string) => void entries.delete(key)
    }
  };
  return entries;
}


const quality = (overrides: Partial<CalibrationQuality> = {}): CalibrationQuality => ({
  rms_reprojection_px: 0.41,
  band: "GOOD",
  views_used: 18,
  views_submitted: 26,
  diagonal_fov_degrees: 72.7,
  fov_plausible: true,
  advice: "Calibration is good. This phone will not need calibrating again.",
  ...overrides
});


// getDeviceId is stable across calls and persists.
{
  const entries = installLocalStorage();
  const first = getDeviceId();
  const second = getDeviceId();
  assert(first === second, "getDeviceId returned a different id on the second call.");
  assert(first.length > 0, "getDeviceId returned an empty id.");
  assert(
    entries.get("cricvision.deviceId") === first,
    "getDeviceId did not persist the id it returned."
  );

  setDeviceLabel("  Raghav's Pixel  ");
  assert(getDeviceLabel() === "Raghav's Pixel", "The device label was not trimmed and stored.");
  setDeviceLabel("   ");
  assert(getDeviceLabel() === null, "A blank label should clear the stored one.");
}

// An implausible field of view renders no numbers, only the advice.
{
  const advice = "The solved field of view is outside the range a phone main camera produces.";
  const display = describeQuality(
    quality({ fov_plausible: false, diagonal_fov_degrees: 43.3, advice })
  );
  assert(display.rms === "—", `Expected an em dash for the error, got ${display.rms}.`);
  assert(
    display.diagonalFov === "—",
    `Expected an em dash for the field of view, got ${display.diagonalFov}.`
  );
  assert(display.fovWarning !== null, "An implausible field of view must carry a warning.");
  assert(display.advice === advice, "The service advice must be passed through verbatim.");
}

// A POOR band never gets to quote a numeric quality claim.
{
  const display = describeQuality(quality({ band: "POOR", rms_reprojection_px: 4.82 }));
  assert(display.rms === "—", `A POOR band quoted an error of ${display.rms}.`);
  assert(display.bandTone === "warn", "A POOR band must read as a failure.");
  assert(
    !JSON.stringify(display).includes("4.82"),
    "The POOR reprojection error leaked into the rendered output."
  );
}

// A good solve does show its numbers.
{
  const display = describeQuality(quality());
  assert(display.rms === "0.41", `Expected 0.41, got ${display.rms}.`);
  assert(display.views === "18 of 26", `Expected "18 of 26", got ${display.views}.`);
  assert(display.diagonalFov === "72.7°", `Expected 72.7°, got ${display.diagonalFov}.`);
  assert(display.fovWarning === null, "A plausible field of view must not warn.");
}

console.log("device calibration display and identity checks passed");
