"use client";

import Image from "next/image";
import { type ChangeEvent, useMemo, useRef, useState } from "react";

import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { DraggableGuideBoxes } from "@/components/live/DraggableGuideBoxes";
import { CAMERA_ALIGNMENT_BOXES, UPLOAD_ALIGNMENT_BOXES } from "@/components/live/StumpAlignmentOverlay";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { runBatPoseCalibration, type BatDetectorModelKey, type ManualBatBoxes } from "@/lib/api";
import type { BatPoseCalibrationResponse, BoxLayout, CapturedFrame, NormalizedBox } from "@/lib/types";


type Source = "camera" | "upload";
type Stage = "idle" | "solving" | "done";

const DEFAULT_BAT_HEIGHT_M = 0.86;

function generateSessionAnalysisId(): string {
  const now = new Date();
  const pad = (value: number, length = 2) => String(value).padStart(length, "0");
  const date = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}`;
  const time = `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  const hex = Array.from({ length: 6 }, () => "0123456789abcdef"[Math.floor(Math.random() * 16)]).join("");
  return `analysis_${date}_${time}_${hex}`;
}

function readUploadedFrame(file: File): Promise<CapturedFrame> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read that image."));
    reader.onload = () => {
      if (typeof reader.result !== "string") {
        reject(new Error("Could not read that image."));
        return;
      }
      const image = new window.Image();
      image.onerror = () => reject(new Error("Choose a valid JPG, PNG, or WebP image."));
      image.onload = () => resolve({ dataUrl: reader.result as string, width: image.naturalWidth, height: image.naturalHeight });
      image.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

function statusTone(result: BatPoseCalibrationResponse | null): "good" | "warn" | "neutral" {
  if (!result) return "neutral";
  if (result.success && result.calibration?.mode === "METRIC_3D") return "good";
  return "warn";
}

function toPixelBox(box: NormalizedBox, frameWidth: number, frameHeight: number) {
  return {
    x: Math.round(box.x * frameWidth),
    y: Math.round(box.y * frameHeight),
    width: Math.round(box.width * frameWidth),
    height: Math.round(box.height * frameHeight)
  };
}

function DetectionRow({ label, found, confidence }: { label: string; found?: boolean; confidence?: number }) {
  return (
    <div className="rounded-lg bg-black/20 p-3">
      <p className="text-xs capitalize text-white/45">{label}</p>
      <p className={`mt-1 text-sm font-bold ${found ? "text-lime" : "text-[#ffaaa6]"}`}>
        {found ? `${((confidence ?? 0) * 100).toFixed(1)}%` : "Not found"}
      </p>
    </div>
  );
}

export default function BatCalibrationPage() {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const analysisId = useMemo(() => generateSessionAnalysisId(), []);

  const [source, setSource] = useState<Source>("camera");
  const [stage, setStage] = useState<Stage>("idle");
  const [batHeightM, setBatHeightM] = useState(DEFAULT_BAT_HEIGHT_M);
  // ponytail: CricShot10k's bat model is trained on in-swing batting footage,
  // not a bat standing static/upright — it finds nothing on this setup, so
  // the drawn box is used directly as the detection by default.
  const [useManualBoxes, setUseManualBoxes] = useState(true);
  const [batDetectorModelKey, setBatDetectorModelKey] = useState<BatDetectorModelKey>("cricshot10k");
  const [cameraBoxLayout, setCameraBoxLayout] = useState<BoxLayout>(CAMERA_ALIGNMENT_BOXES);
  const [uploadBoxLayout, setUploadBoxLayout] = useState<BoxLayout>(UPLOAD_ALIGNMENT_BOXES);
  const [uploadedFrame, setUploadedFrame] = useState<CapturedFrame | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [result, setResult] = useState<BatPoseCalibrationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const frame = await readUploadedFrame(file);
      setUploadedFrame(frame);
      setUploadedFileName(file.name);
      setResult(null);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read that image.");
    } finally {
      event.target.value = "";
    }
  }

  async function runDetection() {
    const frame = source === "camera" ? cameraRef.current?.captureFrame() : uploadedFrame;
    if (!frame) {
      setError(source === "upload" ? "Upload a photo with a bat visible at each end." : "Camera frame is not ready. Wait a moment and try again.");
      return;
    }
    setError(null);
    setResult(null);
    setStage("solving");
    try {
      const layout = source === "camera" ? cameraBoxLayout : uploadBoxLayout;
      const manualBoxes: ManualBatBoxes | null = useManualBoxes
        ? {
            striker: toPixelBox(layout.striker, frame.width, frame.height),
            non_striker: toPixelBox(layout.non_striker, frame.width, frame.height)
          }
        : null;
      const response = await runBatPoseCalibration(analysisId, frame, layout, batHeightM, manualBoxes, batDetectorModelKey);
      setResult(response);
      setStage("done");
    } catch (caught) {
      setError(caught instanceof Error && caught.message !== "Failed to fetch" ? caught.message : "Bat-pose calibration backend is unavailable.");
      setStage("idle");
    }
  }

  const statusMessage =
    stage === "solving" ? (useManualBoxes ? "Solving camera pose from your boxes..." : "Detecting bats and solving camera pose...")
    : result ? result.message
    : useManualBoxes
      ? "Drag each box tightly around its bat, then press Solve."
      : "Stand one bat at each end, on the pitch centreline, then Detect.";

  return (
    <div className="mx-auto max-w-4xl py-6">
      <StatusBadge label="No stumps needed" tone="warn" />
      <h1 className="mt-4 text-3xl font-black tracking-[-0.03em]">Bat Calibration <span className="text-lime">(fallback)</span></h1>
      <p className="mt-3 max-w-2xl text-sm leading-6 text-white/55">
        Solves camera pose from two bats instead of stumps, using an assumed bat length
        rather than measuring your bat. Confidence is capped below the stump calibration
        path — treat results here as a lower-accuracy stand-in for testing, not a
        replacement for real stumps.
      </p>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <div className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1.5">
          <Button className="px-3 py-2" variant={source === "camera" ? "primary" : "secondary"} onClick={() => { setSource("camera"); setResult(null); setError(null); }}>
            Camera
          </Button>
          <Button className="px-3 py-2" variant={source === "upload" ? "primary" : "secondary"} onClick={() => { setSource("upload"); setResult(null); setError(null); }}>
            Upload Photo
          </Button>
        </div>
        <div className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1.5">
          <Button className="px-3 py-2" variant={useManualBoxes ? "primary" : "secondary"} onClick={() => { setUseManualBoxes(true); setResult(null); setError(null); }}>
            Use my box
          </Button>
          <Button className="px-3 py-2" variant={!useManualBoxes ? "primary" : "secondary"} onClick={() => { setUseManualBoxes(false); setResult(null); setError(null); }}>
            Try auto-detect
          </Button>
        </div>
      </div>
      {useManualBoxes ? (
        <p className="mt-2 text-xs text-white/40">
          The bat model doesn&apos;t recognise a static upright bat well, so your dragged box is used
          directly as the bat&apos;s location — position it as tightly as you can around each bat.
        </p>
      ) : (
        <div className="mt-3 flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.1em] text-white/45">Detector model</span>
          <div className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1.5">
            <Button className="px-3 py-2" variant={batDetectorModelKey === "cricshot10k" ? "primary" : "secondary"} onClick={() => setBatDetectorModelKey("cricshot10k")}>
              CricShot10k
            </Button>
            <Button className="px-3 py-2" variant={batDetectorModelKey === "bat_v2" ? "primary" : "secondary"} onClick={() => setBatDetectorModelKey("bat_v2")}>
              Bat v2 (new)
            </Button>
          </div>
        </div>
      )}

      <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_20rem]">
        <div>
          <div className="relative">
            {source === "camera" ? (
              <CameraPreview ref={cameraRef} />
            ) : (
              <div className="relative aspect-video overflow-hidden rounded-2xl border border-white/10 bg-black">
                {uploadedFrame ? (
                  <Image src={uploadedFrame.dataUrl} alt="Uploaded bat calibration photo" fill unoptimized className="object-contain" />
                ) : (
                  <div className="absolute inset-0 grid place-items-center p-8 text-center">
                    <div>
                      <p className="text-lg font-bold">Upload a photo showing one bat at each end.</p>
                      <p className="mt-2 text-sm text-white/45">Both bats upright, standing on the pitch centreline.</p>
                    </div>
                  </div>
                )}
              </div>
            )}
            {(source === "camera" || uploadedFrame) && (
              <DraggableGuideBoxes
                layout={source === "camera" ? cameraBoxLayout : uploadBoxLayout}
                onChange={source === "camera" ? setCameraBoxLayout : setUploadBoxLayout}
                editable={stage !== "solving"}
              />
            )}
          </div>
          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="text-xs text-white/40">Drag each box onto its bat. Corners resize.</p>
            <button
              type="button"
              className="text-xs font-bold text-lime hover:text-white"
              onClick={() => (source === "camera" ? setCameraBoxLayout(CAMERA_ALIGNMENT_BOXES) : setUploadBoxLayout(UPLOAD_ALIGNMENT_BOXES))}
            >
              Reset boxes
            </button>
          </div>
        </div>
        <div className="space-y-4">
          <Card>
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/45">Bat length (m)</p>
            <input
              className="mt-2 w-full rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-sm font-bold text-white"
              type="number"
              step="0.01"
              min="0.3"
              max="1.2"
              value={batHeightM}
              onChange={(event) => setBatHeightM(Number(event.target.value) || DEFAULT_BAT_HEIGHT_M)}
            />
            <p className="mt-2 text-[11px] leading-4 text-white/40">
              Standard adult bat is ~0.86m. Wrong length biases every 3D measurement (speed, swing, bounce).
            </p>
          </Card>

          {source === "upload" && (
            <Card>
              <label htmlFor="bat-calibration-image" className="block cursor-pointer rounded-xl border border-dashed border-white/20 bg-white/5 px-4 py-3 text-center text-sm font-bold transition hover:bg-white/10">
                {uploadedFrame ? "Choose another photo" : "Choose photo"}
              </label>
              <input id="bat-calibration-image" className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => void handleUpload(event)} />
              {uploadedFileName && <p className="mt-2 truncate text-center text-xs text-white/40">{uploadedFileName}</p>}
            </Card>
          )}

          <Card>
            <StatusBadge label={error ? "Failed" : statusMessage} tone={error ? "warn" : statusTone(result)} />
            {error && <p className="mt-3 text-sm text-[#ffaaa6]">{error}</p>}

            {result?.detections && (
              <div className="mt-4 grid grid-cols-2 gap-3">
                <DetectionRow label="Far end (striker)" found={result.detections.striker?.found} confidence={result.detections.striker?.confidence} />
                <DetectionRow label="Near end (non-striker)" found={result.detections.non_striker?.found} confidence={result.detections.non_striker?.confidence} />
              </div>
            )}

            {result?.calibration && (
              <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-4 text-sm">
                <p><span className="text-white/45">Mode:</span> <span className="font-bold">{result.calibration.mode}</span></p>
                <p className="mt-1"><span className="text-white/45">Confidence:</span> <span className="font-bold">{result.calibration.confidence} ({(result.calibration.calibration_confidence * 100).toFixed(0)}%)</span></p>
                {result.calibration.reprojection_error_px != null && (
                  <p className="mt-1"><span className="text-white/45">Reprojection RMSE:</span> <span className="font-bold">{result.calibration.reprojection_error_px.toFixed(2)}px</span></p>
                )}
                {result.calibration.warnings.length > 0 && (
                  <ul className="mt-3 space-y-1 text-xs text-white/45">
                    {result.calibration.warnings.map((warning) => (
                      <li key={warning}>&bull; {warning}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <Button className="mt-5 w-full" disabled={stage === "solving" || (source === "upload" && !uploadedFrame)} onClick={() => void runDetection()}>
              {stage === "solving" ? "Solving..." : result ? "Retry" : useManualBoxes ? "Solve" : "Detect"}
            </Button>
          </Card>
        </div>
      </div>
    </div>
  );
}
