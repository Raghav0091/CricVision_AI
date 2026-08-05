import {
  clampNativePoint,
  defaultWicketBoxes,
  normalizedPointToNative,
  normalizedToWicketBox,
  wicketBoxToNormalized
} from "./coordinates";


function close(actual: number, expected: number, tolerance = 0.001) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Expected ${actual} to be within ${tolerance} of ${expected}.`);
  }
}


const defaults = defaultWicketBoxes(1920, 1080, 12);
close(defaults.NEAR.x, 422.4);
close(defaults.NEAR.y, 518.4);
close(defaults.NEAR.width, 1075.2);
close(defaults.NEAR.height, 518.4);
close(defaults.FAR.y, 172.8);

const roundTrip = wicketBoxToNormalized(
  normalizedToWicketBox(
    { x: 0.2, y: 0.35, width: 0.18, height: 0.22 },
    "FAR",
    1280,
    720,
    4
  )
);
close(roundTrip.x, 0.2);
close(roundTrip.y, 0.35);
close(roundTrip.width, 0.18);
close(roundTrip.height, 0.22);

const native = normalizedPointToNative({ x: 0.5, y: 0.75 }, 3840, 2160);
close(native.x, 1920);
close(native.y, 1620);

const clamped = clampNativePoint({ x: -10, y: 5000 }, 1280, 720);
close(clamped.x, 0);
close(clamped.y, 720);

const corrected = clampNativePoint({ x: 640.4, y: 360.6 }, 1280, 720);
close(corrected.x, 640.4);
close(corrected.y, 360.6);
