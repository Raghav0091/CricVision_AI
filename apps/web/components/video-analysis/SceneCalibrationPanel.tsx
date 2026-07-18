"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  confirmVideoAnalysisCalibration,
  detectVideoAnalysisCalibration,
  type CalibrationV2Result,
  type ConfirmedVideoCalibrationResponse,
  type VideoAnalysisPreparedResponse,
  type WicketCalibration
} from "@/lib/api";

import {
  CalibrationCanvas,
  calculateApproximatePitchGeometry,
  wicketDistanceWarning,
  wicketFromBox
} from "./CalibrationCanvas";
import { CalibrationV2Panel } from "./CalibrationV2Panel";


type CalibrationPhase = "idle" | "detecting" | "editing" | "saving" | "saved";
type DetectedPositions = {
  striker: WicketCalibration | null;
  nonStriker: WicketCalibration | null;
};


export function SceneCalibrationPanel({
  analysis,
  initialCalibration,
  initialCalibrationV2,
  onCalibrated,
  onCalibratedV2,
  onDirty
}: {
  analysis: VideoAnalysisPreparedResponse;
  initialCalibration: ConfirmedVideoCalibrationResponse | null;
  initialCalibrationV2: CalibrationV2Result | null;
  onCalibrated: (calibration: ConfirmedVideoCalibrationResponse) => void;
  onCalibratedV2: (calibration: CalibrationV2Result) => void;
  onDirty: () => void;
}) {
  const [phase, setPhase] = useState<CalibrationPhase>(
    initialCalibration ? "saved" : "idle"
  );
  const [striker, setStriker] = useState<WicketCalibration | null>(
    initialCalibration?.striker_wicket ?? null
  );
  const [nonStriker, setNonStriker] = useState<WicketCalibration | null>(
    initialCalibration?.non_striker_wicket ?? null
  );
  const [detectedPositions, setDetectedPositions] = useState<DetectedPositions>({
    striker: null,
    nonStriker: null
  });
  const [corridorWidth, setCorridorWidth] = useState(
    initialCalibration?.pitch_geometry.corridor_width_multiplier ?? 1
  );
  const [message, setMessage] = useState(
    initialCalibration?.message
      ?? "Run stump detection or place both wicket boxes manually."
  );
  const [error, setError] = useState<string | null>(null);
  const [backendWarning, setBackendWarning] = useState<string | null>(null);
  const [candidateCount, setCandidateCount] = useState(0);
  const [modelPath, setModelPath] = useState<string | null>(
    initialCalibration?.model_path_used ?? null
  );
  const [imageWidth, setImageWidth] = useState(
    initialCalibration?.image_width ?? analysis.width
  );
  const [imageHeight, setImageHeight] = useState(
    initialCalibration?.image_height ?? analysis.height
  );
  const [referenceUrl, setReferenceUrl] = useState(
    initialCalibration?.reference_frame_url ?? analysis.reference_frame_url
  );
  const [savedCalibration, setSavedCalibration] = useState(initialCalibration);
  const [userNote, setUserNote] = useState(initialCalibration?.user_note ?? "");

  useEffect(() => {
    if (!initialCalibration) return;
    setPhase("saved");
    setStriker(initialCalibration.striker_wicket);
    setNonStriker(initialCalibration.non_striker_wicket);
    setCorridorWidth(
      initialCalibration.pitch_geometry.corridor_width_multiplier
    );
    setMessage(initialCalibration.message);
    setModelPath(initialCalibration.model_path_used ?? null);
    setImageWidth(initialCalibration.image_width);
    setImageHeight(initialCalibration.image_height);
    setReferenceUrl(initialCalibration.reference_frame_url);
    setSavedCalibration(initialCalibration);
    setUserNote(initialCalibration.user_note ?? "");
  }, [initialCalibration]);

  const pitchGeometry = useMemo(
    () => calculateApproximatePitchGeometry(striker, nonStriker, corridorWidth),
    [striker, nonStriker, corridorWidth]
  );
  const proximityWarning = wicketDistanceWarning(striker, nonStriker);
  const warning = proximityWarning ?? backendWarning;
  const bothWicketsReady = striker !== null && nonStriker !== null;

  async function runDetection() {
    setPhase("detecting");
    setError(null);
    setBackendWarning(null);
    try {
      const result = await detectVideoAnalysisCalibration(analysis.analysis_id);
      const nextStriker = result.provisional_striker_wicket ?? null;
      const nextNonStriker = result.provisional_non_striker_wicket ?? null;
      setStriker(nextStriker);
      setNonStriker(nextNonStriker);
      setDetectedPositions({
        striker: nextStriker,
        nonStriker: nextNonStriker
      });
      setCandidateCount(result.candidates.length);
      setModelPath(result.model_path_used);
      setImageWidth(result.image_width);
      setImageHeight(result.image_height);
      setReferenceUrl(result.reference_frame_url);
      setMessage(result.message);
      setBackendWarning(result.warning ?? null);
      setSavedCalibration(null);
      onDirty();
      setPhase("editing");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Stump detection failed.");
      setPhase("editing");
    }
  }

  function addManualWicket(label: "striker" | "non_striker") {
    const box = label === "striker"
      ? { x: 0.46, y: 0.28, width: 0.08, height: 0.18 }
      : { x: 0.39, y: 0.62, width: 0.18, height: 0.28 };
    const wicket = wicketFromBox(label, "manual", null, box);
    if (label === "striker") setStriker(wicket);
    else setNonStriker(wicket);
    setBackendWarning(null);
    setMessage("Manual wicket box added. Move and resize it around the complete wicket set.");
    setSavedCalibration(null);
    onDirty();
    setPhase("editing");
  }

  function changeWicket(
    label: "striker" | "non_striker",
    wicket: WicketCalibration
  ) {
    if (label === "striker") setStriker(wicket);
    else setNonStriker(wicket);
    setBackendWarning(null);
    setSavedCalibration(null);
    onDirty();
    if (phase === "saved") setPhase("editing");
  }

  function swapWicketEnds() {
    if (!striker || !nonStriker) return;
    setStriker(wicketFromBox(
      "striker",
      nonStriker.source,
      nonStriker.confidence ?? null,
      nonStriker.box
    ));
    setNonStriker(wicketFromBox(
      "non_striker",
      striker.source,
      striker.confidence ?? null,
      striker.box
    ));
    setSavedCalibration(null);
    onDirty();
    setPhase("editing");
  }

  function resetDetectedPositions() {
    setStriker(detectedPositions.striker);
    setNonStriker(detectedPositions.nonStriker);
    setCorridorWidth(1);
    setBackendWarning(null);
    setSavedCalibration(null);
    onDirty();
    setPhase("editing");
  }

  function changeCorridorWidth(value: number) {
    const nextValue = Math.max(0.7, Math.min(1.5, Math.round(value * 20) / 20));
    setCorridorWidth(nextValue);
    setSavedCalibration(null);
    onDirty();
    if (phase === "saved") setPhase("editing");
  }

  async function confirmCalibration() {
    if (!striker || !nonStriker) {
      setError("Place both wicket boxes before confirming calibration.");
      return;
    }
    if (proximityWarning) {
      setError(proximityWarning);
      return;
    }
    setPhase("saving");
    setError(null);
    try {
      const confirmed = await confirmVideoAnalysisCalibration(
        analysis.analysis_id,
        {
          analysis_id: analysis.analysis_id,
          striker_wicket: {
            label: "striker",
            source: striker.source,
            confidence: striker.confidence ?? null,
            box: striker.box
          },
          non_striker_wicket: {
            label: "non_striker",
            source: nonStriker.source,
            confidence: nonStriker.confidence ?? null,
            box: nonStriker.box
          },
          corridor_width_multiplier: corridorWidth,
          user_note: userNote.trim() || null
        }
      );
      setStriker(confirmed.striker_wicket);
      setNonStriker(confirmed.non_striker_wicket);
      setCorridorWidth(confirmed.pitch_geometry.corridor_width_multiplier);
      setMessage(confirmed.message);
      setSavedCalibration(confirmed);
      setPhase("saved");
      onCalibrated(confirmed);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Calibration could not be saved.");
      setPhase("editing");
    }
  }

  const hasDetectedReset = detectedPositions.striker !== null
    || detectedPositions.nonStriker !== null;

  return (
    <Card className="border-[#ffca68]/25">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <StatusBadge
            label={phase === "saved" ? "Calibration Confirmed" : "Scene Calibration"}
            tone={phase === "saved" ? "good" : "neutral"}
          />
          <h2 className="mt-4 text-2xl font-black">Calibrate both wicket ends</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-white/55">
            Place one box around each complete wicket set. Adjust the boxes so they tightly contain the three stumps and bails.
          </p>
        </div>
        <Button
          variant="secondary"
          disabled={phase === "detecting" || phase === "saving"}
          onClick={() => void runDetection()}
        >
          {phase === "detecting" ? "Running Stump Detection..." : "Run Stump Detection"}
        </Button>
      </div>

      {error && (
        <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}
      <div className="mt-5 rounded-xl border border-white/10 bg-black/20 p-4">
        <p className="text-sm leading-6 text-white/65">{message}</p>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-white/35">
          <span>{candidateCount} detector candidate{candidateCount === 1 ? "" : "s"}</span>
          {modelPath && <span>Model: {modelPath}</span>}
        </div>
      </div>
      {warning && (
        <p className="mt-4 rounded-xl border border-[#ffca68]/35 bg-[#ffca68]/10 p-4 text-sm text-[#ffe0a3]">
          {warning}
        </p>
      )}

      <div className="mt-6">
        <CalibrationCanvas
          imageUrl={referenceUrl}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          striker={striker}
          nonStriker={nonStriker}
          pitchGeometry={pitchGeometry}
          disabled={phase === "detecting" || phase === "saving"}
          onWicketChange={changeWicket}
        />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div className="grid gap-3 sm:grid-cols-2">
          <WicketSummary
            title="Striker Wicket"
            wicket={striker}
            color="#ffca68"
            onAdd={() => addManualWicket("striker")}
          />
          <WicketSummary
            title="Non-Striker Wicket"
            wicket={nonStriker}
            color="#50dcff"
            onAdd={() => addManualWicket("non_striker")}
          />
        </div>
        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <label htmlFor="corridor-width" className="text-sm font-bold">
            Pitch Corridor Width
          </label>
          <div className="mt-3 flex items-center gap-3">
            <button
              type="button"
              aria-label="Decrease Pitch Corridor Width"
              className="h-9 w-9 shrink-0 rounded-lg border border-white/15 bg-white/5 text-lg font-black hover:bg-white/10 disabled:opacity-40"
              disabled={!bothWicketsReady || phase === "saving" || corridorWidth <= 0.7}
              onClick={() => changeCorridorWidth(corridorWidth - 0.05)}
            >
              −
            </button>
            <input
              id="corridor-width"
              className="w-full accent-[#d5ff6b]"
              type="range"
              min="0.7"
              max="1.5"
              step="0.05"
              value={corridorWidth}
              disabled={!bothWicketsReady || phase === "saving"}
              onChange={(event) => changeCorridorWidth(Number(event.target.value))}
            />
            <button
              type="button"
              aria-label="Increase Pitch Corridor Width"
              className="h-9 w-9 shrink-0 rounded-lg border border-white/15 bg-white/5 text-lg font-black hover:bg-white/10 disabled:opacity-40"
              disabled={!bothWicketsReady || phase === "saving" || corridorWidth >= 1.5}
              onClick={() => changeCorridorWidth(corridorWidth + 0.05)}
            >
              +
            </button>
            <strong className="w-12 text-right text-sm text-lime">
              {corridorWidth.toFixed(2)}×
            </strong>
          </div>
          <p className="mt-3 text-xs leading-5 text-white/35">
            Approximate 2D geometry only. Confirm that the trapezoid visually covers the pitch.
          </p>
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <Button
          variant="secondary"
          disabled={!bothWicketsReady || phase === "saving"}
          onClick={swapWicketEnds}
        >
          Swap Wicket Ends
        </Button>
        <Button
          variant="secondary"
          disabled={!hasDetectedReset || phase === "saving"}
          onClick={resetDetectedPositions}
        >
          Reset to Detected Positions
        </Button>
      </div>

      <label htmlFor="calibration-note" className="mt-6 block text-sm font-bold">
        Calibration note <span className="font-normal text-white/35">(optional)</span>
      </label>
      <textarea
        id="calibration-note"
        className="mt-2 min-h-20 w-full rounded-xl border border-white/10 bg-black/25 p-3 text-sm outline-none transition focus:border-lime/40"
        maxLength={1000}
        value={userNote}
        placeholder="Record any manual adjustment or camera setup detail."
        onChange={(event) => {
          setUserNote(event.target.value);
          setSavedCalibration(null);
          onDirty();
          if (phase === "saved") setPhase("editing");
        }}
      />

      <Button
        className="mt-5 w-full sm:w-auto"
        disabled={!bothWicketsReady || Boolean(proximityWarning) || phase === "saving"}
        onClick={() => void confirmCalibration()}
      >
        {phase === "saving"
          ? "Saving Scene Calibration..."
          : phase === "saved"
            ? "Save Calibration Changes"
            : "Confirm Scene Calibration"}
      </Button>

      {savedCalibration && (
        <div className="mt-7 rounded-xl border border-lime/20 bg-lime/[0.04] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="font-black text-lime">Scene calibration confirmed</p>
              <p className="mt-1 text-xs text-white/40">
                Axis uses the bottom-centre of both wicket boxes.
              </p>
            </div>
            <a
              className="text-xs font-bold text-lime underline"
              href={savedCalibration.calibration_url}
              target="_blank"
              rel="noreferrer"
            >
              Open calibration JSON
            </a>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="mt-4 h-auto w-full rounded-lg bg-black object-contain"
            src={`${savedCalibration.calibration_overlay_url}?v=${encodeURIComponent(savedCalibration.updated_at)}`}
            alt="Confirmed approximate 2D calibration overlay"
          />
        </div>
      )}

      <CalibrationV2Panel
        analysis={analysis}
        initialCalibration={initialCalibrationV2}
        onCalibrated={onCalibratedV2}
      />
    </Card>
  );
}


function WicketSummary({
  title,
  wicket,
  color,
  onAdd
}: {
  title: string;
  wicket: WicketCalibration | null;
  color: string;
  onAdd: () => void;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-4">
      <p className="font-black" style={{ color }}>{title}</p>
      {wicket ? (
        <>
          <p className="mt-2 text-xs uppercase tracking-[0.1em] text-white/35">
            {wicket.source}
            {wicket.confidence != null && ` · ${(wicket.confidence * 100).toFixed(1)}% confidence`}
          </p>
          <p className="mt-3 font-mono text-[11px] leading-5 text-white/45">
            x {wicket.box.x.toFixed(3)} · y {wicket.box.y.toFixed(3)}
            <br />
            w {wicket.box.width.toFixed(3)} · h {wicket.box.height.toFixed(3)}
          </p>
        </>
      ) : (
        <Button className="mt-3 w-full" variant="secondary" onClick={onAdd}>
          Add {title} Box
        </Button>
      )}
    </div>
  );
}
