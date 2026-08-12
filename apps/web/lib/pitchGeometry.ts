/** The pitch the operator says they are actually standing on.
 *
 * Registration solves "where must the camera be for this pitch to look like
 * this". Ask it about a 20.12 m pitch while the rig is three metres of indoor
 * floor and no camera pose satisfies it, so every candidate is rejected. The
 * geometry declared here is what makes an improvised rig solvable at all.
 *
 * `null` everywhere means regulation. That keeps the normal case the default
 * and leaves every existing full-size flow byte-for-byte unchanged.
 */

import type { CricketPitchGeometry } from "@/lib/wicketCalibration/types";


// Mirrors packages/cricket_vision/calibration/cricket_pitch_geometry.py.
export const REGULATION_PITCH: CricketPitchGeometry = {
  pitch_length_m: 20.12,
  wicket_width_m: 0.2286,
  wicket_height_m: 0.7112,
  stump_diameter_m: 0.0381,
  pitch_width_m: 3.05,
  popping_crease_distance_m: 1.22
};

// The backend bounds these too; clamping here just means the operator cannot
// send a request that is certain to 422.
export const MIN_PITCH_LENGTH_M = 0.5;
export const MAX_PITCH_LENGTH_M = 40;

const STORAGE_KEY = "cricvision.pitchGeometry";


function storage(): Storage | null {
  // Server rendering has no localStorage, and neither does a browser with
  // storage blocked. Both should fall back to regulation, not throw.
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}


function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}


/** True when this geometry is regulation and can be sent as `null`. */
export function isRegulationPitch(geometry: CricketPitchGeometry | null): boolean {
  if (geometry === null) return true;
  return (
    geometry.pitch_length_m === REGULATION_PITCH.pitch_length_m &&
    geometry.wicket_width_m === REGULATION_PITCH.wicket_width_m &&
    geometry.wicket_height_m === REGULATION_PITCH.wicket_height_m
  );
}


/**
 * Coerce partial operator input into a geometry the backend will accept.
 *
 * The popping crease has to stay inside half the pitch, so a short rig gets a
 * proportionally short crease rather than the regulation 1.22 m that would be
 * past the far wicket.
 */
export function normalizePitchGeometry(
  input: Partial<CricketPitchGeometry>
): CricketPitchGeometry {
  const length = clamp(
    Number.isFinite(input.pitch_length_m) ? (input.pitch_length_m as number) : REGULATION_PITCH.pitch_length_m,
    MIN_PITCH_LENGTH_M,
    MAX_PITCH_LENGTH_M
  );
  const wicketWidth = clamp(
    Number.isFinite(input.wicket_width_m) ? (input.wicket_width_m as number) : REGULATION_PITCH.wicket_width_m,
    0.02,
    1
  );
  const wicketHeight = clamp(
    Number.isFinite(input.wicket_height_m) ? (input.wicket_height_m as number) : REGULATION_PITCH.wicket_height_m,
    0.05,
    2
  );
  // The backend rejects a stump diameter at or past half the wicket width.
  const stumpDiameter = clamp(
    Number.isFinite(input.stump_diameter_m) ? (input.stump_diameter_m as number) : REGULATION_PITCH.stump_diameter_m,
    0.001,
    wicketWidth / 2 - 0.001
  );
  const pitchWidth = clamp(
    Number.isFinite(input.pitch_width_m) ? (input.pitch_width_m as number) : REGULATION_PITCH.pitch_width_m,
    wicketWidth,
    10
  );
  const creaseCeiling = Math.min(5, length / 2 - 0.001);
  const crease = clamp(
    Number.isFinite(input.popping_crease_distance_m)
      ? (input.popping_crease_distance_m as number)
      : Math.min(REGULATION_PITCH.popping_crease_distance_m, creaseCeiling),
    0.001,
    creaseCeiling
  );
  return {
    pitch_length_m: length,
    wicket_width_m: wicketWidth,
    wicket_height_m: wicketHeight,
    stump_diameter_m: stumpDiameter,
    pitch_width_m: pitchWidth,
    popping_crease_distance_m: crease
  };
}


/** Read the declared pitch. `null` means regulation. */
export function loadPitchGeometry(): CricketPitchGeometry | null {
  const raw = storage()?.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<CricketPitchGeometry>;
    const normalized = normalizePitchGeometry(parsed);
    return isRegulationPitch(normalized) ? null : normalized;
  } catch {
    // A corrupt entry should read as "regulation", never crash the setup screen.
    return null;
  }
}


/** Persist the declared pitch so a test rig survives a reload. */
export function savePitchGeometry(geometry: CricketPitchGeometry | null): void {
  const store = storage();
  if (!store) return;
  if (geometry === null || isRegulationPitch(geometry)) {
    store.removeItem(STORAGE_KEY);
    return;
  }
  store.setItem(STORAGE_KEY, JSON.stringify(normalizePitchGeometry(geometry)));
}


/**
 * One short line naming the pitch a reading was measured on.
 *
 * A 3.2 m rig reporting 14 km/h has to be obviously a rig reading, never
 * mistakable for a net reading, so this returns a label for anything that is
 * not regulation and `null` for the normal case.
 */
export function describePitchGeometry(
  geometry: CricketPitchGeometry | null
): string | null {
  if (isRegulationPitch(geometry)) return null;
  const declared = geometry as CricketPitchGeometry;
  const parts = [`${declared.pitch_length_m.toFixed(2)} m pitch`];
  if (declared.wicket_height_m !== REGULATION_PITCH.wicket_height_m) {
    parts.push(`${Math.round(declared.wicket_height_m * 100)} cm stumps`);
  }
  if (declared.wicket_width_m !== REGULATION_PITCH.wicket_width_m) {
    parts.push(`${Math.round(declared.wicket_width_m * 100)} cm wicket`);
  }
  return parts.join(" · ");
}
