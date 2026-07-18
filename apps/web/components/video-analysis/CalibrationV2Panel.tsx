"use client";

import { useEffect, useMemo, useState } from "react";

import { CalibrationV2LandmarkEditor } from "./CalibrationV2LandmarkEditor";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  confirmVideoAnalysisCalibrationV2,
  initialiseVideoAnalysisCalibrationV2,
  type CalibrationLandmarkInput,
  type CalibrationV2InitialiseResponse,
  type CalibrationV2Result,
  type CricketPitchGeometry,
  type ImageLeftRightConvention,
  type VideoAnalysisPreparedResponse
} from "@/lib/api";


type EditorState = {
  referenceFrameUrl: string;
  imageWidth: number;
  imageHeight: number;
  pitchGeometry: CricketPitchGeometry;
  landmarks: CalibrationLandmarkInput[];
  imageConvention: ImageLeftRightConvention;
  warnings: string[];
};


function stateFromInitialised(
  initialised: CalibrationV2InitialiseResponse
): EditorState {
  return {
    referenceFrameUrl: initialised.reference_frame_url,
    imageWidth: initialised.image_width,
    imageHeight: initialised.image_height,
    pitchGeometry: initialised.pitch_geometry,
    landmarks: initialised.landmarks,
    imageConvention: initialised.image_left_right_convention,
    warnings: initialised.warnings
  };
}


function stateFromSaved(saved: CalibrationV2Result): EditorState {
  return {
    referenceFrameUrl: saved.reference_frame_url,
    imageWidth: saved.image_width,
    imageHeight: saved.image_height,
    pitchGeometry: saved.pitch_geometry,
    landmarks: [
      ...saved.landmark_set.primary_stump_bases,
      ...saved.landmark_set.optional_ground_landmarks
    ],
    imageConvention: saved.coordinate_system.image_left_right_convention,
    warnings: saved.quality.warnings
  };
}


function changedSource(
  source: CalibrationLandmarkInput["source"]
): CalibrationLandmarkInput["source"] {
  return source === "manual" ? "manual" : "manually_adjusted";
}


function matchingId(
  landmark: CalibrationLandmarkInput,
  wicketEnd: "bowler" | "striker"
): string {
  return landmark.id.replace(/^(bowler|striker)_/, `${wicketEnd}_`);
}


function swappedSideId(landmark: CalibrationLandmarkInput): string {
  if (landmark.id.includes("_left_")) {
    return landmark.id.replace("_left_", "_right_");
  }
  if (landmark.id.includes("_right_")) {
    return landmark.id.replace("_right_", "_left_");
  }
  return landmark.id;
}


export function CalibrationV2Panel({
  analysis,
  initialCalibration,
  onCalibrated
}: {
  analysis: VideoAnalysisPreparedResponse;
  initialCalibration: CalibrationV2Result | null;
  onCalibrated: (calibration: CalibrationV2Result) => void;
}) {
  const [editor, setEditor] = useState<EditorState | null>(
    initialCalibration ? stateFromSaved(initialCalibration) : null
  );
  const [autoGuesses, setAutoGuesses] = useState<EditorState | null>(null);
  const [saved, setSaved] = useState<CalibrationV2Result | null>(
    initialCalibration
  );
  const [initialising, setInitialising] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [userNote, setUserNote] = useState(initialCalibration?.user_note ?? "");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialCalibration) {
      const restored = stateFromSaved(initialCalibration);
      setEditor(restored);
      setSaved(initialCalibration);
      setUserNote(initialCalibration.user_note ?? "");
      return;
    }
    void initialise();
    // The analysis ID is stable for the lifetime of this panel.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis.analysis_id, initialCalibration]);

  async function initialise() {
    setInitialising(true);
    setError(null);
    try {
      const initialised = await initialiseVideoAnalysisCalibrationV2(
        analysis.analysis_id
      );
      const next = stateFromInitialised(initialised);
      setEditor(next);
      setAutoGuesses(next);
      setSaved(null);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Calibration v2 could not be initialised."
      );
    } finally {
      setInitialising(false);
    }
  }

  function updateLandmark(
    landmarkId: string,
    normalizedX: number,
    normalizedY: number
  ) {
    setEditor((current) => current ? {
      ...current,
      landmarks: current.landmarks.map((landmark) => (
        landmark.id === landmarkId
          ? {
              ...landmark,
              normalized_x: normalizedX,
              normalized_y: normalizedY,
              source: changedSource(landmark.source),
              confidence: landmark.source === "manual" ? landmark.confidence : null
            }
          : landmark
      ))
    } : current);
    setSaved(null);
  }

  function swapWicketEnds() {
    setEditor((current) => {
      if (!current) return current;
      const byId = new Map(current.landmarks.map((landmark) => [
        landmark.id,
        landmark
      ]));
      return {
        ...current,
        landmarks: current.landmarks.map((landmark) => {
          if (landmark.wicket_end !== "bowler" && landmark.wicket_end !== "striker") {
            return landmark;
          }
          const oppositeEnd = landmark.wicket_end === "bowler"
            ? "striker"
            : "bowler";
          const opposite = byId.get(matchingId(landmark, oppositeEnd));
          if (!opposite) return landmark;
          return {
            ...landmark,
            normalized_x: opposite.normalized_x,
            normalized_y: opposite.normalized_y,
            source: changedSource(opposite.source),
            confidence: null
          };
        })
      };
    });
    setSaved(null);
  }

  function swapLeftRight() {
    setEditor((current) => {
      if (!current) return current;
      const byId = new Map(current.landmarks.map((landmark) => [
        landmark.id,
        landmark
      ]));
      return {
        ...current,
        imageConvention: current.imageConvention === "image_left_is_world_left"
          ? "image_left_is_world_right"
          : "image_left_is_world_left",
        landmarks: current.landmarks.map((landmark) => {
          const opposite = byId.get(swappedSideId(landmark));
          if (!opposite || opposite.id === landmark.id) return landmark;
          return {
            ...landmark,
            normalized_x: opposite.normalized_x,
            normalized_y: opposite.normalized_y,
            source: changedSource(opposite.source),
            confidence: null
          };
        })
      };
    });
    setSaved(null);
  }

  function resetAutoGuesses() {
    if (autoGuesses) {
      setEditor({
        ...autoGuesses,
        landmarks: autoGuesses.landmarks.map((landmark) => ({ ...landmark }))
      });
      setSaved(null);
      return;
    }
    void initialise();
  }

  function changePitchGeometry(
    key: keyof CricketPitchGeometry,
    value: number
  ) {
    if (!Number.isFinite(value)) return;
    setEditor((current) => current ? {
      ...current,
      pitchGeometry: {
        ...current.pitchGeometry,
        [key]: value
      }
    } : current);
    setSaved(null);
  }

  async function confirmCalibration() {
    if (!editor) return;
    setSaving(true);
    setError(null);
    try {
      const result = await confirmVideoAnalysisCalibrationV2(
        analysis.analysis_id,
        {
          analysis_id: analysis.analysis_id,
          landmarks: editor.landmarks.map((landmark) => ({
            id: landmark.id,
            label: landmark.label,
            wicket_end: landmark.wicket_end,
            landmark_type: landmark.landmark_type,
            normalized_x: landmark.normalized_x,
            normalized_y: landmark.normalized_y,
            source: landmark.source,
            confidence: landmark.confidence ?? null,
            world_x_m: landmark.landmark_type === "ground_control"
              ? landmark.world_x_m
              : null,
            world_y_m: landmark.landmark_type === "ground_control"
              ? landmark.world_y_m
              : null,
            world_z_m: landmark.landmark_type === "ground_control"
              ? 0
              : null
          })),
          pitch_geometry: editor.pitchGeometry,
          image_left_right_convention: editor.imageConvention,
          user_note: userNote.trim() || null
        }
      );
      setSaved(result);
      setEditor(stateFromSaved(result));
      onCalibrated(result);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Calibration v2 could not be saved."
      );
    } finally {
      setSaving(false);
    }
  }

  const primaryLandmarks = editor?.landmarks.filter(
    (landmark) => landmark.landmark_type === "stump_base"
  ) ?? [];
  const manualCount = primaryLandmarks.filter(
    (landmark) => (
      landmark.source === "manual"
      || landmark.source === "manually_adjusted"
    )
  ).length;
  const quality = saved?.quality ?? null;
  const qualityTone = (
    quality?.quality_grade === "excellent"
    || quality?.quality_grade === "good"
    || quality?.quality_grade === "usable"
  ) ? "good" : quality ? "warn" : "neutral";
  const qualityLabel = quality
    ? quality.quality_grade.replaceAll("_", " ")
    : "Not calculated";

  const worldStumps = useMemo(() => {
    const geometry = editor?.pitchGeometry;
    if (!geometry) return [];
    return [
      { x: 12, y: 50, label: "Bowler" },
      { x: 88, y: 50, label: "Striker" }
    ];
  }, [editor?.pitchGeometry]);

  return (
    <section className="mt-8 border-t border-white/10 pt-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <StatusBadge
            label={saved ? "Calibration v2" : "Ground Plane Setup"}
            tone={saved?.status === "confirmed" ? "good" : "neutral"}
          />
          <h2 className="mt-4 text-2xl font-black">
            Calibration v2 — Pitch world geometry
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-white/55">
            Confirm six stump bases to map image pixels onto the pitch ground
            plane. This does not estimate airborne ball height or a full 3D
            camera pose.
          </p>
        </div>
        <Button
          variant="secondary"
          disabled={initialising || saving}
          onClick={() => void initialise()}
        >
          {initialising ? "Initialising..." : "Reinitialise Landmarks"}
        </Button>
      </div>

      {error && (
        <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}

      <ol className="mt-5 grid gap-2 text-sm text-white/55 sm:grid-cols-2 lg:grid-cols-5">
        {[
          "Check bowler-end wicket.",
          "Check striker-end wicket.",
          "Adjust the six stump-base points.",
          "Confirm calibration.",
          "Inspect projected pitch alignment."
        ].map((instruction, index) => (
          <li
            key={instruction}
            className="rounded-xl border border-white/10 bg-black/20 p-3"
          >
            <span className="mr-2 font-black text-lime">{index + 1}.</span>
            {instruction}
          </li>
        ))}
      </ol>

      {initialising && !editor && (
        <p className="mt-6 rounded-xl border border-lime/20 bg-lime/[0.04] p-4 text-sm text-lime">
          Initialising stump-base guesses from the existing wicket calibration…
        </p>
      )}

      {editor && (
        <>
          <div className="mt-7">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">
                  1–3 · Reference, detected wickets, stump landmarks
                </p>
                <p className="mt-2 text-sm text-[#ffe0a3]">
                  Automatic landmarks are approximate. Adjust stump-base
                  markers before confirming.
                </p>
              </div>
              <button
                type="button"
                className="text-xs font-bold text-lime underline"
                onClick={() => setShowLabels((current) => !current)}
              >
                {showLabels ? "Hide Labels" : "Show Labels"}
              </button>
            </div>
            <div className="mt-4">
              <CalibrationV2LandmarkEditor
                imageUrl={editor.referenceFrameUrl}
                imageWidth={editor.imageWidth}
                imageHeight={editor.imageHeight}
                landmarks={editor.landmarks}
                disabled={saving}
                showLabels={showLabels}
                onLandmarkChange={updateLandmark}
              />
            </div>
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-xs text-white/40">
              <span>Six required landmarks: {primaryLandmarks.length}/6</span>
              <span>Manually adjusted: {manualCount}/6</span>
              <span>
                Image convention:{" "}
                {editor.imageConvention === "image_left_is_world_left"
                  ? "image left = world left"
                  : "image left = world right"}
              </span>
            </div>
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-3">
            <Button
              variant="secondary"
              disabled={saving}
              onClick={resetAutoGuesses}
            >
              Reset Auto Guesses
            </Button>
            <Button
              variant="secondary"
              disabled={saving}
              onClick={swapWicketEnds}
            >
              Swap Wicket Ends
            </Button>
            <Button
              variant="secondary"
              disabled={saving}
              onClick={swapLeftRight}
            >
              Swap Left / Right
            </Button>
          </div>

          <div className="mt-7 grid gap-5 lg:grid-cols-2">
            <div className="rounded-xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">
                4 · World geometry preview
              </p>
              <svg
                className="mt-4 h-auto w-full rounded-lg bg-[#132112]"
                viewBox="0 0 100 46"
                role="img"
                aria-label="Top-down cricket pitch world geometry preview"
              >
                <rect
                  x="12"
                  y="8"
                  width="76"
                  height="30"
                  fill="none"
                  stroke="#d5ff6b"
                  strokeWidth="0.7"
                />
                <line x1="12" y1="23" x2="88" y2="23" stroke="white" strokeDasharray="2 1" />
                <line x1="17" y1="8" x2="17" y2="38" stroke="#50dcff" />
                <line x1="83" y1="8" x2="83" y2="38" stroke="#ffca68" />
                {worldStumps.map((wicket) => (
                  <g key={wicket.label}>
                    <line
                      x1={wicket.x}
                      y1="20.8"
                      x2={wicket.x}
                      y2="25.2"
                      stroke={wicket.label === "Bowler" ? "#50dcff" : "#ffca68"}
                      strokeWidth="1.4"
                    />
                    <text
                      x={wicket.x}
                      y="43"
                      fill="white"
                      fontSize="3"
                      textAnchor="middle"
                    >
                      {wicket.label}
                    </text>
                  </g>
                ))}
              </svg>
              <p className="mt-3 text-xs leading-5 text-white/40">
                Origin: bowler wicket centre. +X points toward striker, +Y is
                lateral, +Z is upward. All dimensions are metres.
              </p>
            </div>

            <div className="rounded-xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">
                Pitch geometry
              </p>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {([
                  ["pitch_length_m", "Pitch length"],
                  ["wicket_width_m", "Wicket width"],
                  ["wicket_height_m", "Wicket height"],
                  ["pitch_width_m", "Pitch width"],
                  ["popping_crease_distance_m", "Popping crease"]
                ] as const).map(([key, label]) => (
                  <label key={key} className="text-xs text-white/45">
                    {label} (m)
                    <input
                      className="mt-1 w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-white outline-none focus:border-lime/40"
                      type="number"
                      min="0.01"
                      step="0.0001"
                      value={editor.pitchGeometry[key]}
                      disabled={saving}
                      onChange={(event) => changePitchGeometry(
                        key,
                        Number(event.target.value)
                      )}
                    />
                  </label>
                ))}
              </div>
              <p className="mt-3 text-xs leading-5 text-white/35">
                Defaults are regulation adult dimensions; competition and
                junior formats can differ.
              </p>
            </div>
          </div>

          {editor.warnings.length > 0 && !saved && (
            <div className="mt-5 rounded-xl border border-[#ffca68]/25 bg-[#ffca68]/[0.06] p-4 text-xs leading-5 text-[#ffe0a3]">
              {editor.warnings.map((warning) => (
                <p key={warning}>• {warning}</p>
              ))}
            </div>
          )}

          <label htmlFor="calibration-v2-note" className="mt-6 block text-sm font-bold">
            Calibration v2 note{" "}
            <span className="font-normal text-white/35">(optional)</span>
          </label>
          <textarea
            id="calibration-v2-note"
            className="mt-2 min-h-20 w-full rounded-xl border border-white/10 bg-black/25 p-3 text-sm outline-none transition focus:border-lime/40"
            maxLength={1000}
            value={userNote}
            disabled={saving}
            placeholder="Record camera orientation or manual landmark details."
            onChange={(event) => {
              setUserNote(event.target.value);
              setSaved(null);
            }}
          />

          <Button
            className="mt-5 w-full sm:w-auto"
            disabled={saving || primaryLandmarks.length !== 6}
            onClick={() => void confirmCalibration()}
          >
            {saving ? "Calculating Ground Homography..." : "Confirm Calibration v2"}
          </Button>
        </>
      )}

      {saved && (
        <div className="mt-8 rounded-xl border border-lime/20 bg-lime/[0.04] p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <StatusBadge label={qualityLabel} tone={qualityTone} />
                <p className="font-black">
                  {saved.status === "confirmed"
                    ? "Ground-plane transform available"
                    : "Insufficient geometry"}
                </p>
              </div>
              <p className="mt-2 text-sm text-white/50">{saved.message}</p>
            </div>
            <a
              className="text-xs font-bold text-lime underline"
              href={saved.calibration_v2_url}
              target="_blank"
              rel="noreferrer"
            >
              Open calibration_v2.json
            </a>
          </div>

          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <QualityMetric
              label="Reprojection RMSE"
              value={quality?.reprojection_rmse_px != null
                ? `${quality.reprojection_rmse_px.toFixed(2)} px`
                : "Unavailable"}
            />
            <QualityMetric
              label="Max error"
              value={quality?.max_reprojection_error_px != null
                ? `${quality.max_reprojection_error_px.toFixed(2)} px`
                : "Unavailable"}
            />
            <QualityMetric
              label="Geometry"
              value={quality?.geometry_condition.replaceAll("_", " ") ?? "Unavailable"}
            />
            <QualityMetric
              label="Image coverage"
              value={quality ? `${(quality.image_coverage * 100).toFixed(2)}%` : "Unavailable"}
            />
          </div>

          {quality && quality.warnings.length > 0 && (
            <div className="mt-4 rounded-lg border border-[#ffca68]/20 bg-black/20 p-3 text-xs leading-5 text-[#ffe0a3]">
              {quality.warnings.map((warning) => (
                <p key={warning}>• {warning}</p>
              ))}
            </div>
          )}

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-white/40">
                Reference frame
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                className="mt-3 h-auto w-full rounded-lg bg-black object-contain"
                src={saved.reference_frame_url}
                alt="Calibration v2 clean reference frame"
              />
            </div>
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-white/40">
                Authoritative Calibration v2 overlay
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                className="mt-3 h-auto w-full rounded-lg bg-black object-contain"
                src={`${saved.calibration_v2_overlay_url}?v=${encodeURIComponent(saved.updated_at)}`}
                alt="Calibration v2 ground-plane reprojection overlay"
              />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}


function QualityMetric({
  label,
  value
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
      <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
        {label}
      </p>
      <p className="mt-2 text-sm font-black capitalize">{value}</p>
    </div>
  );
}
