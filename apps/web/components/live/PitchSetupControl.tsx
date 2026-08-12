"use client";

import { useEffect, useState } from "react";

import {
  MAX_PITCH_LENGTH_M,
  MIN_PITCH_LENGTH_M,
  REGULATION_PITCH,
  isRegulationPitch,
  normalizePitchGeometry,
  savePitchGeometry
} from "@/lib/pitchGeometry";
import type { CricketPitchGeometry } from "@/lib/wicketCalibration/types";


function NumberField({
  label,
  hint,
  value,
  min,
  max,
  onChange
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  /** Accepted for call-site readability; the text field does its own stepping. */
  step?: number;
  onChange: (next: number) => void;
}) {
  // The field keeps its own text while focused. A fully controlled numeric
  // input cannot be typed into here: the parent clamps on every keystroke and
  // writes the result back, so an empty field snaps to the old value and a
  // trailing "." is deleted the moment it is typed — making any decimal
  // impossible to enter. Text in, number out, reconcile on blur.
  const [text, setText] = useState(() => String(value));
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!editing) setText(String(value));
  }, [value, editing]);

  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-bold uppercase tracking-wide text-white/60">{label}</span>
      <input
        type="text"
        inputMode="decimal"
        value={text}
        onFocus={() => setEditing(true)}
        onChange={(event) => {
          setText(event.target.value);
          const next = Number.parseFloat(event.target.value);
          if (Number.isFinite(next)) onChange(next);
        }}
        onBlur={() => {
          setEditing(false);
          const parsed = Number.parseFloat(text);
          // Show what was actually stored, which may have been clamped.
          setText(String(Number.isFinite(parsed) ? value : value));
        }}
        aria-label={label}
        className="w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm tabular-nums text-white outline-none focus:border-white/40"
      />
      <span className="text-[10px] text-white/40">
        {hint} · {min}–{max}
      </span>
    </label>
  );
}


/**
 * Let the operator declare the pitch before calibrating.
 *
 * 22 yards is the normal case and stays the default. The custom fields exist
 * because an improvised indoor rig is neither 20.12 m long nor fitted with
 * full-size stumps, and getting those wrong scales every distance the replay
 * later reports.
 */
export function PitchSetupControl({
  geometry,
  onChange
}: {
  /** `null` means regulation. */
  geometry: CricketPitchGeometry | null;
  onChange: (next: CricketPitchGeometry | null) => void;
}) {
  const custom = !isRegulationPitch(geometry);
  const [draft, setDraft] = useState<CricketPitchGeometry>(
    geometry ?? REGULATION_PITCH
  );

  // The declared pitch is read from localStorage after mount, so the initial
  // state above misses it. Without this the restored rig would show regulation
  // numbers in the fields while sending the stored ones.
  useEffect(() => {
    if (geometry) setDraft(geometry);
  }, [geometry]);

  function apply(next: CricketPitchGeometry) {
    const normalized = normalizePitchGeometry(next);
    setDraft(normalized);
    const value = isRegulationPitch(normalized) ? null : normalized;
    savePitchGeometry(value);
    onChange(value);
  }

  function toggleCustom(enabled: boolean) {
    if (!enabled) {
      setDraft(REGULATION_PITCH);
      savePitchGeometry(null);
      onChange(null);
      return;
    }
    // Seed a plausible short rig rather than regulation, so switching to
    // "custom" and leaving it alone does not silently mean 22 yards.
    apply({ ...REGULATION_PITCH, pitch_length_m: 4, popping_crease_distance_m: 1 });
  }

  return (
    <section className="rounded-xl border border-white/15 bg-black/45 p-4 backdrop-blur-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold text-white">Pitch setup</h3>
          <p className="mt-0.5 text-[11px] text-white/50">
            {custom ? "Custom rig — readings apply to this pitch only." : "Full 22 yards (20.12 m)."}
          </p>
        </div>
        <label className="flex shrink-0 items-center gap-2 text-[11px] font-semibold text-white/70">
          <input
            type="checkbox"
            checked={custom}
            onChange={(event) => toggleCustom(event.target.checked)}
            className="h-4 w-4 accent-[#ffcf4d]"
          />
          Custom pitch
        </label>
      </div>

      {custom && (
        <div className="mt-4 grid grid-cols-2 gap-3">
          <NumberField
            label="Pitch length"
            hint={`metres, ${MIN_PITCH_LENGTH_M}–${MAX_PITCH_LENGTH_M}`}
            value={draft.pitch_length_m}
            min={MIN_PITCH_LENGTH_M}
            max={MAX_PITCH_LENGTH_M}
            step={0.1}
            onChange={(pitch_length_m) => apply({ ...draft, pitch_length_m })}
          />
          <NumberField
            label="Stump height"
            hint="metres, top of stump"
            value={draft.wicket_height_m}
            min={0.05}
            max={2}
            step={0.01}
            onChange={(wicket_height_m) => apply({ ...draft, wicket_height_m })}
          />
          <NumberField
            label="Wicket width"
            hint="metres, outside to outside"
            value={draft.wicket_width_m}
            min={0.02}
            max={1}
            step={0.01}
            onChange={(wicket_width_m) => apply({ ...draft, wicket_width_m })}
          />
          <NumberField
            label="Pitch width"
            hint="metres, across the strip"
            value={draft.pitch_width_m}
            min={draft.wicket_width_m}
            max={10}
            step={0.05}
            onChange={(pitch_width_m) => apply({ ...draft, pitch_width_m })}
          />
        </div>
      )}
    </section>
  );
}
