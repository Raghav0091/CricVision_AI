"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getDeviceCalibration } from "@/lib/api";
import type { DeviceLensProfile } from "@/lib/deviceCalibration/types";
import { getDeviceId } from "@/lib/deviceIdentity";


/** Says whether the numbers about to be produced rest on a measured lens.
 *
 * Worth knowing before bowling rather than after: an estimated lens can still
 * return a confident-looking speed and bounce point that are simply wrong,
 * because focal length and camera distance trade against each other.
 */
export function LensCalibrationBadge({ className = "" }: { className?: string }) {
  const [profile, setProfile] = useState<DeviceLensProfile | null>(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    getDeviceCalibration(getDeviceId())
      .then(setProfile)
      .catch(() => setProfile(null))
      .finally(() => setChecked(true));
  }, []);

  // Claiming either state before the lookup returns would be a guess.
  if (!checked) return null;

  const calibrated = profile !== null && profile.quality.fov_plausible;
  if (calibrated) {
    return (
      <span
        className={`inline-flex items-center gap-2 rounded-full border border-lime/30 bg-lime/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] text-lime ${className}`}
      >
        Lens calibrated
        <span className="font-mono tabular-nums normal-case tracking-normal text-lime/70">
          {profile.quality.rms_reprojection_px.toFixed(2)} px
        </span>
      </span>
    );
  }

  return (
    <Link
      href="/calibrate-device"
      className={`inline-flex items-center gap-2 rounded-full border border-[#ffe761]/30 bg-[#ffe761]/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] text-[#ffe761] transition hover:bg-[#ffe761]/20 ${className}`}
    >
      Lens estimated
      <span className="normal-case tracking-normal underline">Calibrate</span>
    </Link>
  );
}
