"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  deleteDeviceCalibration,
  getDeviceCalibration,
  solveDeviceCalibration
} from "@/lib/api";
import { describeQuality } from "@/lib/deviceCalibration/quality";
import type { DeviceLensProfile } from "@/lib/deviceCalibration/types";
import { getDeviceId, getDeviceLabel, setDeviceLabel } from "@/lib/deviceIdentity";


type Stage = "intro" | "recording" | "solving" | "result";

// Long enough to sweep tilt, distance and all four corners without the file
// growing past what a phone tunnel will carry.
const GUIDE_SECONDS = 30;

const CHECKLIST = [
  ["Tilt the board", "Hold it at 20–45° to the camera, not flat-on. Straight-on views cannot separate focal length from distance."],
  ["Reach the corners", "Take the board into all four corners of the frame. Distortion only shows up at the edges."],
  ["Vary distance", "Close enough to fill the frame, far enough to be small."],
  ["Move slowly", "Motion blur destroys corner precision, and corner precision is the ceiling on the whole result."]
] as const;


function supportedMimeType(): string | undefined {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
}


export default function CalibrateDevicePage() {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [stage, setStage] = useState<Stage>("intro");
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<DeviceLensProfile | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [label, setLabel] = useState("");
  const [columns, setColumns] = useState(9);
  const [rows, setRows] = useState(6);
  const [squareSizeMm, setSquareSizeMm] = useState(25);

  useEffect(() => {
    setLabel(getDeviceLabel() ?? "");
    getDeviceCalibration(getDeviceId())
      .then((stored) => {
        if (!stored) return;
        setProfile(stored);
        setColumns(stored.checkerboard.columns);
        setRows(stored.checkerboard.rows);
        setSquareSizeMm(stored.checkerboard.square_size_mm);
        setStage("result");
      })
      .catch(() => {
        // A missing backend should leave the page usable for a fresh solve
        // rather than blocking on a lookup that was only ever advisory.
      });
  }, []);

  useEffect(() => {
    if (stage !== "recording") return;
    const start = Date.now();
    const interval = window.setInterval(() => setElapsedSeconds((Date.now() - start) / 1000), 200);
    return () => window.clearInterval(interval);
  }, [stage]);

  const solve = useCallback(
    async (blob: Blob) => {
      setStage("solving");
      setError(null);
      try {
        if (label.trim()) setDeviceLabel(label);
        const response = await solveDeviceCalibration(
          blob,
          getDeviceId(),
          label.trim() || getDeviceLabel(),
          { columns, rows, square_size_mm: squareSizeMm }
        );
        setProfile(response.profile);
        setStage("result");
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Calibration failed.");
        setStage("intro");
      }
    },
    [columns, label, rows, squareSizeMm]
  );

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  function startRecording() {
    const stream = cameraRef.current?.getStream();
    if (!stream) {
      setError("The camera is not ready yet.");
      return;
    }
    const mimeType = supportedMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
      void solve(blob);
    };
    recorderRef.current = recorder;
    recorder.start();
    setElapsedSeconds(0);
    setError(null);
    setStage("recording");
  }

  async function recalibrate() {
    setError(null);
    try {
      await deleteDeviceCalibration(getDeviceId());
    } catch {
      // A profile that will not delete is still about to be overwritten by the
      // next successful solve, so this should not block re-filming.
    }
    setProfile(null);
    setStage("intro");
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-black">Lens Calibration</h1>
          <p className="text-sm text-white/60">
            Solve this phone&apos;s focal length and distortion once. Every later camera
            pose stops guessing them.
          </p>
        </div>
        {profile && (
          <StatusBadge
            label={profile.quality.fov_plausible ? "Lens calibrated" : "Lens unusable"}
            tone={profile.quality.fov_plausible ? "good" : "warn"}
          />
        )}
      </header>

      {error && (
        <Card className="border-signal/30 bg-signal/5">
          <p className="text-sm text-[#ffaaa6]">{error}</p>
        </Card>
      )}

      {/* One preview element, kept mounted across intro and recording:
          remounting it would reopen the camera and orphan the MediaRecorder
          that is attached to the stream it is about to discard. */}
      {(stage === "intro" || stage === "recording") && (
        <Card>
          <CameraPreview ref={cameraRef} />
        </Card>
      )}

      {stage === "intro" && (
        <Card className="space-y-5">
          <div className="space-y-2 text-sm text-white/70">
            <h2 className="text-base font-bold text-white">Before you start</h2>
            <p>
              Print a chessboard pattern and tape it to something rigid — a clipboard or
              a sheet of card. A board that bows even slightly will corrupt the solve,
              and nothing downstream will tell you it happened.
            </p>
            <p>
              Count <strong className="text-white">inner corners</strong>, not squares. A
              10×7 chessboard has 9×6 inner corners, which is what the defaults below
              describe.
            </p>
            <p>
              Measure one square with a ruler{" "}
              <strong className="text-white">after printing</strong>. Printers rescale to
              fit the page, so the number on the PDF is usually not the number on the
              paper — and this measurement is what puts the result in millimetres.
            </p>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="text-xs font-bold uppercase tracking-[0.12em] text-white/50">
              Name for this phone
              <input
                value={label}
                onChange={(event) => setLabel(event.target.value)}
                placeholder="Raghav's Pixel"
                className="mt-1 w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm font-normal normal-case tracking-normal text-white"
              />
            </label>
            <label className="text-xs font-bold uppercase tracking-[0.12em] text-white/50">
              Square size (mm, measured)
              <input
                type="number"
                min={2}
                max={200}
                step={0.5}
                value={squareSizeMm}
                onChange={(event) => setSquareSizeMm(Number(event.target.value))}
                className="mt-1 w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm font-normal tracking-normal text-white"
              />
            </label>
            <label className="text-xs font-bold uppercase tracking-[0.12em] text-white/50">
              Inner corners across
              <input
                type="number"
                min={3}
                max={20}
                value={columns}
                onChange={(event) => setColumns(Number(event.target.value))}
                className="mt-1 w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm font-normal tracking-normal text-white"
              />
            </label>
            <label className="text-xs font-bold uppercase tracking-[0.12em] text-white/50">
              Inner corners down
              <input
                type="number"
                min={3}
                max={20}
                value={rows}
                onChange={(event) => setRows(Number(event.target.value))}
                className="mt-1 w-full rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-sm font-normal tracking-normal text-white"
              />
            </label>
          </div>
          {columns === rows && (
            <p className="text-sm text-[#ffaaa6]">
              Use a board with different corner counts. A square grid can be read at four
              different rotations, so corner order flips between views and the solve
              degrades without saying so.
            </p>
          )}

          <Button onClick={startRecording} disabled={columns === rows}>
            Start recording
          </Button>
        </Card>
      )}

      {stage === "recording" && (
        <Card className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <p className="font-mono text-2xl font-black tabular-nums">
              {elapsedSeconds.toFixed(0)}s
              <span className="ml-2 text-sm font-normal text-white/45">
                of about {GUIDE_SECONDS}s
              </span>
            </p>
            <Button variant="camera" onClick={stopRecording}>
              Stop and solve
            </Button>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-lime transition-[width]"
              style={{ width: `${Math.min(100, (elapsedSeconds / GUIDE_SECONDS) * 100)}%` }}
            />
          </div>
          <ul className="space-y-2">
            {CHECKLIST.map(([title, detail]) => (
              <li key={title} className="rounded-xl border border-white/10 bg-white/5 p-3">
                <p className="text-sm font-bold text-white">{title}</p>
                <p className="text-xs text-white/60">{detail}</p>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {stage === "solving" && (
        <Card className="flex items-center gap-3">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-lime" />
          <p className="text-sm text-white/70">
            Finding the board in every frame and solving the lens…
          </p>
        </Card>
      )}

      {stage === "result" && profile && (
        <ResultView profile={profile} onRecalibrate={recalibrate} />
      )}

      {stage === "result" && !profile && (
        <Card>
          <p className="text-sm text-white/70">No profile was produced.</p>
          <Button className="mt-3" variant="secondary" onClick={() => setStage("intro")}>
            Try again
          </Button>
        </Card>
      )}
    </div>
  );
}


function ResultView({
  profile,
  onRecalibrate
}: {
  profile: DeviceLensProfile;
  onRecalibrate: () => void;
}) {
  const display = describeQuality(profile.quality);
  const tone = { good: "text-lime", amber: "text-[#ffe761]", warn: "text-[#ffaaa6]" }[display.bandTone];
  const chip = {
    good: "border-lime/30 bg-lime/10 text-lime",
    amber: "border-[#ffe761]/30 bg-[#ffe761]/10 text-[#ffe761]",
    warn: "border-signal/30 bg-signal/10 text-[#ffaaa6]"
  }[display.bandTone];

  return (
    <Card className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <span
          className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] ${chip}`}
        >
          {display.bandLabel}
        </span>
        <p className="text-sm text-white/60">
          {profile.device_label || "This device"} ·{" "}
          {new Date(profile.calibrated_at).toLocaleDateString()}
        </p>
      </div>

      {display.fovWarning && <p className={`text-sm ${tone}`}>{display.fovWarning}</p>}

      <dl className="grid gap-4 sm:grid-cols-3">
        <Metric label="Reprojection error" value={display.rms} suffix=" px" />
        <Metric label="Views used" value={display.views} />
        <Metric label="Diagonal field of view" value={display.diagonalFov} />
      </dl>

      <p className="text-sm text-white/70">{display.advice}</p>

      <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-sm text-white/60">
        <p>
          Measured at{" "}
          <strong className="text-white">
            {profile.image_width}×{profile.image_height}
          </strong>
          , focal length{" "}
          <strong className="text-white">{profile.focal_length_x_px.toFixed(0)} px</strong>.
        </p>
        <p className="mt-2">
          Film cricket at this same shape of frame. The focal length scales cleanly to a
          higher or lower resolution, but not across a different aspect ratio — switching
          between landscape and portrait, or turning on a crop mode, makes this profile
          inapplicable and the solver will go back to guessing.
        </p>
      </div>

      <Button variant="secondary" onClick={onRecalibrate}>
        Re-calibrate
      </Button>
    </Card>
  );
}


function Metric({ label, value, suffix = "" }: { label: string; value: string; suffix?: string }) {
  return (
    <div>
      <dt className="text-xs font-bold uppercase tracking-[0.12em] text-white/45">{label}</dt>
      <dd className="mt-1 font-mono text-xl font-black tabular-nums">
        {value}
        {value === "—" ? "" : suffix}
      </dd>
    </div>
  );
}
