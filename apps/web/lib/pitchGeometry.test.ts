import {
  MAX_PITCH_LENGTH_M,
  MIN_PITCH_LENGTH_M,
  REGULATION_PITCH,
  describePitchGeometry,
  isRegulationPitch,
  loadPitchGeometry,
  normalizePitchGeometry,
  savePitchGeometry
} from "./pitchGeometry";


function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function close(actual: number, expected: number, tolerance = 1e-9) {
  if (Math.abs(actual - expected) > tolerance) {
    throw new Error(`Expected ${actual} to be within ${tolerance} of ${expected}.`);
  }
}


// A minimal localStorage so the module under test can run outside a browser.
class MemoryStorage {
  private entries = new Map<string, string>();
  get length() {
    return this.entries.size;
  }
  getItem(key: string) {
    return this.entries.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.entries.set(key, String(value));
  }
  removeItem(key: string) {
    this.entries.delete(key);
  }
  clear() {
    this.entries.clear();
  }
  key(index: number) {
    return Array.from(this.entries.keys())[index] ?? null;
  }
}

const store = new MemoryStorage();
(globalThis as { window?: unknown }).window = { localStorage: store };


// Regulation is the default and must round-trip as "nothing declared", so an
// unchanged full-size setup keeps sending null exactly as it always has.
assert(isRegulationPitch(null), "null must read as regulation.");
assert(isRegulationPitch(REGULATION_PITCH), "Regulation constants must read as regulation.");
assert(loadPitchGeometry() === null, "An empty store must read as regulation.");

savePitchGeometry(REGULATION_PITCH);
assert(loadPitchGeometry() === null, "Saving regulation must clear the override.");


// Test 8: custom geometry round-trips through localStorage.
const rig = normalizePitchGeometry({
  pitch_length_m: 3.2,
  wicket_height_m: 0.4,
  wicket_width_m: 0.12,
  stump_diameter_m: 0.02
});
savePitchGeometry(rig);

const restored = loadPitchGeometry();
assert(restored !== null, "A declared rig must survive a reload.");
close(restored!.pitch_length_m, 3.2);
close(restored!.wicket_height_m, 0.4);
close(restored!.wicket_width_m, 0.12);
close(restored!.stump_diameter_m, 0.02);
assert(!isRegulationPitch(restored), "A 3.2 m rig is not regulation.");

savePitchGeometry(null);
assert(loadPitchGeometry() === null, "Clearing must return to regulation.");


// A short pitch cannot keep the regulation 1.22 m popping crease, because the
// backend rejects a crease at or beyond half the pitch length.
const short = normalizePitchGeometry({ pitch_length_m: 2 });
assert(
  short.popping_crease_distance_m < short.pitch_length_m / 2,
  "The popping crease must stay inside half the declared pitch."
);

// Lengths are clamped to the range the backend accepts.
close(normalizePitchGeometry({ pitch_length_m: 0 }).pitch_length_m, MIN_PITCH_LENGTH_M);
close(normalizePitchGeometry({ pitch_length_m: 999 }).pitch_length_m, MAX_PITCH_LENGTH_M);

// A stump can never be half the wicket width or wider.
const narrow = normalizePitchGeometry({ wicket_width_m: 0.1, stump_diameter_m: 0.09 });
assert(
  narrow.stump_diameter_m < narrow.wicket_width_m / 2,
  "Stump diameter must stay under half the wicket width."
);

// A corrupt entry must read as regulation rather than crash the setup screen.
store.setItem("cricvision.pitchGeometry", "{not json");
assert(loadPitchGeometry() === null, "Corrupt storage must fall back to regulation.");
store.clear();


// Test 9 support: the review screen labels anything that is not regulation, so
// a rig reading can never be mistaken for a net reading.
assert(describePitchGeometry(null) === null, "Regulation needs no rig label.");
assert(
  describePitchGeometry(REGULATION_PITCH) === null,
  "Regulation geometry needs no rig label."
);

const label = describePitchGeometry(rig);
assert(label !== null, "A declared rig must be labelled on the review screen.");
assert(label!.includes("3.20 m pitch"), `Expected the declared length in ${label}.`);
assert(label!.includes("40 cm stumps"), `Expected the declared stump height in ${label}.`);

const lengthOnly = describePitchGeometry(
  normalizePitchGeometry({ pitch_length_m: 4 })
);
assert(
  lengthOnly === "4.00 m pitch",
  `A length-only rig should read as its length, got ${lengthOnly}.`
);

console.log("pitchGeometry.test.ts passed");
