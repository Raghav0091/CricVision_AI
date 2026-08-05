import Link from "next/link";

import type { ReactNode } from "react";



import {

  formatReprojectionStats,

  geometryRejectionLabel

} from "@/lib/virtual-pitch-replay/validatePayload";

import type { MeasurementValidity, ReplayPayloadV1 } from "@/lib/virtual-pitch-replay/types";



function bannerTone(validity: MeasurementValidity): string {

  switch (validity) {

    case "IMAGE_SPACE_ONLY":

      return "border-[#ffca68]/35 bg-[#ffca68]/[0.08] text-[#ffdc9a]";

    case "VISUALIZATION_ONLY":

      return "border-[#ffe761]/35 bg-[#ffe761]/[0.08] text-[#ffe9a8]";

    case "INSUFFICIENT_EVIDENCE":

      return "border-signal/35 bg-signal/10 text-[#ffaaa6]";

    default:

      return "border-white/15 bg-white/[0.04] text-white/70";

  }

}



function geometryFailureMessage(payload: ReplayPayloadV1): string | null {

  const geometry = payload.diagnostics.geometry_validity;

  if (geometry == null || geometry === "VALID_METRIC_3D") {

    return null;

  }

  if (payload.diagnostics.unavailable_reason) {

    return payload.diagnostics.unavailable_reason;

  }

  if (geometry === "OUTSIDE_PITCH_GEOMETRY") {

    return "Calibrated 3D trajectory points fall outside the virtual pitch bounds.";

  }

  if (geometry === "INVALID_REPROJECTION") {

    return "Calibrated 3D trajectory failed bidirectional reprojection against the primary track.";

  }

  return "Calibrated 3D trajectory is unavailable for this analysis.";

}



function trackingReplayHref(analysisId: string): string {

  return `/video-analysis?analysis_id=${encodeURIComponent(analysisId)}`;

}



export function ReplayDegradedBanner({

  payload,

  schemaIssues,

  analysisId

}: {

  payload: ReplayPayloadV1;

  schemaIssues: string[];

  analysisId: string;

}) {

  const banners: ReactNode[] = [];

  const geometryReason = geometryFailureMessage(payload);

  const geometryLabel = geometryRejectionLabel(payload);

  const reprojectionStats = geometryLabel ? formatReprojectionStats(payload) : [];

  const showImageSpaceBanner =

    payload.measurement_validity === "IMAGE_SPACE_ONLY" || geometryReason != null;



  if (schemaIssues.length > 0) {

    banners.push(

      <div

        key="schema"

        className="rounded-lg border border-signal/35 bg-signal/10 px-3 py-2 text-sm leading-6 text-[#ffaaa6]"

        role="status"

      >

        Replay payload schema is not fully supported: {schemaIssues.join(" ")}

      </div>

    );

  }



  if (showImageSpaceBanner) {

    banners.push(

      <div

        key="image-space"

        className={`rounded-lg border px-3 py-2 text-sm leading-6 ${bannerTone("IMAGE_SPACE_ONLY")}`}

        role="status"

      >

        {geometryLabel ? (

          <p className="mb-2 text-[11px] font-bold uppercase tracking-[0.14em] text-[#ffaaa6]">

            {geometryLabel.replaceAll("_", " ")}

          </p>

        ) : null}

        <p>

          {geometryReason

            ?? "World measurements are unavailable for this analysis. The pitch is shown without a measured 3D ball path."}

        </p>

        {reprojectionStats.length > 0 ? (

          <ul className="mt-2 list-none space-y-0.5 text-xs text-white/70">

            {reprojectionStats.map((line) => (

              <li key={line}>{line}</li>

            ))}

          </ul>

        ) : null}

        <p className="mt-2 text-xs text-white/70">

          The 2D Tracking Replay with image-space ball positions remains available.

        </p>

        <Link

          href={trackingReplayHref(analysisId)}

          className="mt-2 inline-block text-xs font-semibold underline underline-offset-2"

        >

          Open 2D Tracking Replay

        </Link>

      </div>

    );

  }



  if (payload.measurement_validity === "VISUALIZATION_ONLY") {

    banners.push(

      <div

        key="visualization"

        className={`rounded-lg border px-3 py-2 text-sm leading-6 ${bannerTone("VISUALIZATION_ONLY")}`}

        role="status"

      >

        Visualization only — camera and trajectory are illustrative. No measured delivery metrics are shown.

      </div>

    );

  }



  if (payload.measurement_validity === "INSUFFICIENT_EVIDENCE") {

    banners.push(

      <div

        key="insufficient"

        className={`rounded-lg border px-3 py-2 text-sm leading-6 ${bannerTone("INSUFFICIENT_EVIDENCE")}`}

        role="status"

      >

        {payload.diagnostics.unavailable_reason ?? "Insufficient evidence for a reliable virtual replay."}

        <Link

          href={trackingReplayHref(analysisId)}

          className="mt-2 inline-block text-xs font-semibold underline underline-offset-2"

        >

          Open 2D Tracking Replay

        </Link>

      </div>

    );

  }



  if (payload.diagnostics.warnings.length > 0) {

    banners.push(

      <div

        key="warnings"

        className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs leading-5 text-white/55"

      >

        {payload.diagnostics.warnings.join(" ")}

      </div>

    );

  }



  if (banners.length === 0) return null;



  return <div className="space-y-2">{banners}</div>;

}


