import {
  containedMediaRect,
  displayToVideoNative,
  videoNativeToDisplay
} from "./calibrationCoordinates";


function close(actual: number, expected: number, tolerance = 1e-6) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Expected ${actual} to be within ${tolerance} of ${expected}.`);
  }
}


function roundTrip(
  native: { width: number; height: number },
  container: { width: number; height: number },
  point: { x: number; y: number },
  transform = { zoom: 1, panX: 0, panY: 0, devicePixelRatio: 1 }
) {
  const displayed = videoNativeToDisplay(point, native, container, transform);
  const restored = displayToVideoNative(displayed, native, container, transform);
  if (!restored) throw new Error("A displayed in-frame point became unavailable.");
  close(restored.x, point.x);
  close(restored.y, point.y);
}


const landscape = containedMediaRect(
  { width: 1920, height: 1080 },
  { width: 390, height: 500 }
);
close(landscape.x, 0);
close(landscape.y, (500 - 390 * 9 / 16) / 2);

const portrait = containedMediaRect(
  { width: 1080, height: 1920 },
  { width: 900, height: 500 }
);
close(portrait.x, (900 - 500 * 9 / 16) / 2);
close(portrait.y, 0);

roundTrip(
  { width: 1920, height: 1080 },
  { width: 390, height: 500 },
  { x: 1370.25, y: 440.75 }
);
roundTrip(
  { width: 1080, height: 1920 },
  { width: 390, height: 620 },
  { x: 240.5, y: 1510.25 }
);
roundTrip(
  { width: 3840, height: 2160 },
  { width: 1440, height: 680 },
  { x: 1900, y: 900 },
  { zoom: 2.25, panX: -85, panY: 44, devicePixelRatio: 2 }
);
roundTrip(
  { width: 1280, height: 720 },
  { width: 820, height: 740 },
  { x: 640, y: 360 },
  { zoom: 1.5, panX: 35, panY: -20, devicePixelRatio: 3 }
);

const sameNativePoint = { x: 315, y: 510 };
for (const container of [
  { width: 390, height: 620 },
  { width: 768, height: 520 },
  { width: 1440, height: 700 }
]) {
  roundTrip({ width: 1080, height: 1920 }, container, sameNativePoint);
}

const outside = displayToVideoNative(
  { x: 10, y: 10 },
  { width: 1920, height: 1080 },
  { width: 390, height: 500 }
);
if (outside !== null) throw new Error("Letterbox coordinates must not be saved.");

