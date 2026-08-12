import type { CalibrationQuality } from "./types";


export type QualityDisplay = {
  bandLabel: string;
  bandTone: "good" | "amber" | "warn";
  /** Reprojection error, or "—" when the solve has not earned the right to quote one. */
  rms: string;
  views: string;
  diagonalFov: string;
  fovWarning: string | null;
  advice: string;
};


const BAND_TONE = {
  GOOD: "good",
  ACCEPTABLE: "amber",
  POOR: "warn"
} as const;


/** Formats a solve result without overclaiming.
 *
 * Two cases must not show numbers. An implausible field of view means the
 * solve converged on a lens the phone does not have, so its error figure
 * describes a fiction. A POOR band means the error is large enough that
 * quoting it to two decimals implies a precision the calibration does not
 * have. In both cases the service's advice is the only honest output, so it
 * is passed through verbatim.
 */
export function describeQuality(quality: CalibrationQuality): QualityDisplay {
  const trustworthy = quality.fov_plausible && quality.band !== "POOR";
  return {
    bandLabel: quality.band,
    bandTone: BAND_TONE[quality.band],
    rms: trustworthy ? quality.rms_reprojection_px.toFixed(2) : "—",
    views: `${quality.views_used} of ${quality.views_submitted}`,
    diagonalFov: quality.fov_plausible
      ? `${quality.diagonal_fov_degrees.toFixed(1)}°`
      : "—",
    fovWarning: quality.fov_plausible
      ? null
      : "The solved field of view is not one a phone main camera produces. "
        + "This profile should not be used; film the board again.",
    advice: quality.advice
  };
}
