"use client";

import Image from "next/image";
import { type ChangeEvent, useRef, useState } from "react";

import { CalibrationStatus } from "@/components/live/CalibrationStatus";
import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { DeliveryCapturePanel } from "@/components/live/DeliveryCapturePanel";
import { ExperimentalDeliveryTest } from "@/components/live/ExperimentalDeliveryTest";
import {
  CAMERA_ALIGNMENT_BOXES,
  StumpAlignmentOverlay,
  UPLOAD_ALIGNMENT_BOXES
} from "@/components/live/StumpAlignmentOverlay";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { solveCalibration } from "@/lib/api";
import type { BoxLayout, CalibrationResponse, CapturedFrame, LiveStage, NormalizedBox } from "@/lib/types";


type LiveMode = "camera" | "upload" | "experimental";


const setupItems = [
  ["Fixed camera", "Use a tripod or stable phone mount."],
  ["Six stumps visible", "Keep both stump sets clearly inside the frame."],
  ["Behind non-striker", "Place the camera behind the non-striker stumps."],
  ["Good lighting", "Avoid shadows, glare, and low-light motion blur."],
  ["Clear view", "Keep players and equipment from blocking the camera."]
];


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


function failureMessage(result: CalibrationResponse, mode: LiveMode): string {
  if (result.status === "stump_detector_missing") {
    return "Stump detector model is missing. Add Models/stump_detector/best.pt.";
  }
  if (result.status === "stumps_not_found") {
    if (mode === "upload") {
      const strikerFound = result.detections?.striker.found === true;
      const nonStrikerFound = result.detections?.non_striker.found === true;
      if (nonStrikerFound && !strikerFound) {
        return "Non-striker stumps detected. Move the striker box over the far stumps and press Redetect.";
      }
      if (strikerFound && !nonStrikerFound) {
        return "Striker stumps detected. Move the non-striker box over the near stumps and press Redetect.";
      }
      return "Stumps not found. Adjust both boxes or upload a clearer image.";
    }
    return "Place real cricket stumps inside both boxes and try again.";
  }
  return result.message;
}


function BoxControls({
  layout,
  onChange,
  onReset
}: {
  layout: BoxLayout;
  onChange: (end: keyof BoxLayout, field: keyof NormalizedBox, value: number) => void;
  onReset: () => void;
}) {
  const controls: Array<{ field: keyof NormalizedBox; label: string; min: number; max: number }> = [
    { field: "x", label: "X", min: 0, max: 96 },
    { field: "y", label: "Y", min: 0, max: 92 },
    { field: "width", label: "Width", min: 4, max: 50 },
    { field: "height", label: "Height", min: 8, max: 50 }
  ];
  return (
    <div className="mt-5 rounded-xl border border-white/10 bg-black/20 p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/45">Detection boxes</p>
        <button type="button" className="text-xs font-bold text-lime hover:text-white" onClick={onReset}>Reset defaults</button>
      </div>
      {(["striker", "non_striker"] as const).map((end) => (
        <div key={end} className="mt-4">
          <p className="text-xs font-bold capitalize text-white/70">{end.replace("_", " ")}</p>
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-3">
            {controls.map(({ field, label, min, max }) => (
              <label key={`${end}-${field}`} className="text-[11px] text-white/45">
                <span className="flex justify-between gap-2"><span>{label}</span><span>{Math.round(layout[end][field] * 100)}%</span></span>
                <input
                  className="mt-1 h-1.5 w-full cursor-pointer accent-lime"
                  type="range"
                  min={min}
                  max={max}
                  step="1"
                  value={Math.round(layout[end][field] * 100)}
                  onChange={(event) => onChange(end, field, Number(event.target.value) / 100)}
                />
              </label>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}


function DetectionSummary({ result }: { result: CalibrationResponse | null }) {
  if (!result?.detections) return null;
  return (
    <div className="mt-5 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/45">Detector evidence</p>
        {result.virtual_stumps && <span className="text-right text-[10px] font-bold uppercase text-[#ffe761]">Estimated from bbox</span>}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        {(["striker", "non_striker"] as const).map((end) => {
          const detection = result.detections?.[end];
          return (
            <div key={end} className="rounded-lg bg-black/20 p-3">
              <p className="text-xs capitalize text-white/45">{end.replace("_", " ")}</p>
              <p className={`mt-1 text-sm font-bold ${detection?.found ? "text-lime" : "text-[#ffaaa6]"}`}>
                {detection?.found ? `${(detection.confidence * 100).toFixed(1)}%` : "Not found"}
              </p>
            </div>
          );
        })}
      </div>
      {result.calibration_quality && (
        <p className="mt-3 text-xs text-white/45">Calibration score {(result.calibration_quality.score * 100).toFixed(1)}%</p>
      )}
    </div>
  );
}


export default function LivePage() {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const [mode, setMode] = useState<LiveMode>("camera");
  const [stage, setStage] = useState<LiveStage>("setup");
  const [message, setMessage] = useState<string | null>(null);
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResponse | null>(null);
  const [frameSize, setFrameSize] = useState<{ width: number; height: number } | null>(null);
  const [uploadedFrame, setUploadedFrame] = useState<CapturedFrame | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadBoxLayout, setUploadBoxLayout] = useState<BoxLayout>(UPLOAD_ALIGNMENT_BOXES);
  const [deliveryCount] = useState(0);

  function enterCalibration(nextMode: LiveMode) {
    setMode(nextMode);
    setMessage(null);
    setCalibrationResult(null);
    setFrameSize(nextMode === "upload" && uploadedFrame ? { width: uploadedFrame.width, height: uploadedFrame.height } : null);
    setStage("align-stumps");
  }

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const frame = await readUploadedFrame(file);
      setUploadedFrame(frame);
      setUploadedFileName(file.name);
      setUploadBoxLayout(UPLOAD_ALIGNMENT_BOXES);
      setFrameSize({ width: frame.width, height: frame.height });
      setCalibrationResult(null);
      setMessage(null);
      setStage("align-stumps");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not read that image.");
    } finally {
      event.target.value = "";
    }
  }

  function updateUploadBox(end: keyof BoxLayout, field: keyof NormalizedBox, value: number) {
    setUploadBoxLayout((current) => {
      const box = current[end];
      const next = { ...box, [field]: value };
      if (field === "x") next.x = Math.min(value, 1 - box.width);
      if (field === "y") next.y = Math.min(value, 1 - box.height);
      if (field === "width") next.width = Math.min(value, 1 - box.x);
      if (field === "height") next.height = Math.min(value, 1 - box.y);
      return { ...current, [end]: next };
    });
  }

  async function continueCalibration() {
    const frame = mode === "camera" ? cameraRef.current?.captureFrame() : uploadedFrame;
    if (!frame) {
      setMessage(mode === "upload" ? "Upload a cricket image to test stump calibration." : "Camera frame is not ready. Wait a moment and try again.");
      return;
    }
    setFrameSize({ width: frame.width, height: frame.height });
    setMessage(null);
    setCalibrationResult(null);
    setStage("solving-calibration");
    try {
      const boxLayout = mode === "upload" ? uploadBoxLayout : CAMERA_ALIGNMENT_BOXES;
      const result = await solveCalibration(frame, boxLayout);
      setCalibrationResult(result);
      const bothStumpSetsFound = result.detections?.striker.found === true && result.detections?.non_striker.found === true;
      if (!result.success || !bothStumpSetsFound) {
        setMessage(result.status === "setup_complete" ? "Stumps not found. Adjust both boxes or upload a clearer image." : failureMessage(result, mode));
        setStage("align-stumps");
        return;
      }
      setStage("setup-complete");
    } catch (error) {
      setMessage(error instanceof Error && error.message !== "Failed to fetch" ? error.message : "Calibration backend is unavailable. Confirm FastAPI is running at the configured API URL.");
      setStage("align-stumps");
    }
  }

  if (stage === "setup") {
    return (
      <div className="mx-auto max-w-6xl py-5">
        <StatusBadge label="Live bowling session" tone="good" />
        <div className="mt-6 grid gap-8 lg:grid-cols-[1.25fr_0.75fr] lg:items-end">
          <div>
            <h1 className="text-5xl font-black tracking-[-0.05em] sm:text-7xl">Set the pitch.<br /><span className="text-lime">Test the calibration.</span></h1>
            <p className="mt-6 max-w-xl text-base leading-7 text-white/55">Use the live camera or upload a cricket image. CricVision only completes setup after both stump sets are detected.</p>
          </div>
          <div className="grid gap-3">
            <Button className="w-full py-4 text-base" onClick={() => enterCalibration("camera")}>Camera Calibration</Button>
            <Button className="w-full py-4 text-base" variant="secondary" onClick={() => enterCalibration("upload")}>Upload Calibration Image</Button>
            <Button className="w-full py-4 text-base" variant="secondary" onClick={() => enterCalibration("experimental")}>Experimental Delivery Test</Button>
          </div>
        </div>
        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {setupItems.map(([title, copy], index) => (
            <Card key={title} className="shadow-none">
              <span className="text-xs font-black text-lime">0{index + 1}</span>
              <h2 className="mt-8 font-bold">{title}</h2>
              <p className="mt-2 text-sm leading-5 text-white/45">{copy}</p>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (mode === "experimental") {
    return <ExperimentalDeliveryTest onSelectCalibration={(nextMode) => enterCalibration(nextMode)} />;
  }

  const showOverlay = stage === "align-stumps" || stage === "solving-calibration" || stage === "setup-complete" || stage === "capturing";
  const previewStyle = mode === "upload" && uploadedFrame ? { aspectRatio: `${uploadedFrame.width} / ${uploadedFrame.height}` } : undefined;

  return (
    <div className="mx-auto max-w-7xl py-2">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-lime">Calibration</p>
          <h1 className="mt-1 text-2xl font-black">{mode === "upload" ? "Upload image setup" : "Camera setup"}</h1>
        </div>
        <div className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1.5">
          <Button className="px-3 py-2" variant={mode === "camera" ? "primary" : "secondary"} onClick={() => enterCalibration("camera")}>Camera Calibration</Button>
          <Button className="px-3 py-2" variant={mode === "upload" ? "primary" : "secondary"} onClick={() => enterCalibration("upload")}>Upload Calibration Image</Button>
          <Button className="px-3 py-2" variant="secondary" onClick={() => enterCalibration("experimental")}>Experimental Delivery Test</Button>
        </div>
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_22rem]">
        <div className="relative">
          {mode === "camera" ? (
            <CameraPreview ref={cameraRef} />
          ) : (
            <div className="relative aspect-video overflow-hidden rounded-2xl border border-white/10 bg-black" style={previewStyle}>
              {uploadedFrame ? (
                <Image src={uploadedFrame.dataUrl} alt="Uploaded cricket calibration" fill unoptimized className="object-contain" />
              ) : (
                <div className="absolute inset-0 grid place-items-center p-8 text-center">
                  <div>
                    <p className="text-lg font-bold">Upload a cricket image to test stump calibration.</p>
                    <p className="mt-2 text-sm text-white/45">Use a clear JPG, PNG, or WebP showing both stump sets.</p>
                  </div>
                </div>
              )}
            </div>
          )}
          {showOverlay && (mode === "camera" || uploadedFrame) && (
            <StumpAlignmentOverlay
              showAlignment={stage === "align-stumps" || stage === "solving-calibration"}
              detections={calibrationResult?.detections}
              virtualStumps={calibrationResult?.virtual_stumps}
              pitchOverlay={calibrationResult?.pitch_overlay}
              boxLayout={mode === "upload" ? uploadBoxLayout : CAMERA_ALIGNMENT_BOXES}
              frameWidth={frameSize?.width}
              frameHeight={frameSize?.height}
              setupComplete={stage === "setup-complete" && calibrationResult?.success === true}
            />
          )}
          {stage === "capturing" && mode === "camera" && <div className="absolute left-4 top-4 rounded-full bg-signal px-3 py-1 text-xs font-black uppercase tracking-wider">Live</div>}
        </div>
        <div className="space-y-4">
          {stage === "align-stumps" && (
            <Card>
              <CalibrationStatus
                status={message ? "Failed" : "Searching"}
                message={message ?? (mode === "upload" ? (uploadedFrame ? "Press Detect to find both stump sets and build the pitch overlay." : "Upload a cricket image to test stump calibration.") : "Fit both stump sets inside the red boxes, then press Continue.")}
              />
              <DetectionSummary result={calibrationResult} />
              {mode === "upload" && uploadedFrame && (
                <BoxControls
                  layout={uploadBoxLayout}
                  onChange={updateUploadBox}
                  onReset={() => {
                    setUploadBoxLayout(UPLOAD_ALIGNMENT_BOXES);
                  }}
                />
              )}
              {mode === "upload" && (
                <div className="mt-5">
                  <label htmlFor="calibration-image" className="block cursor-pointer rounded-xl border border-dashed border-white/20 bg-white/5 px-4 py-3 text-center text-sm font-bold transition hover:bg-white/10">
                    {uploadedFrame ? "Choose another image" : "Choose calibration image"}
                  </label>
                  <input id="calibration-image" className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => void handleUpload(event)} />
                  {uploadedFileName && <p className="mt-2 truncate text-center text-xs text-white/40">{uploadedFileName}</p>}
                </div>
              )}
              <div className="mt-6 grid grid-cols-2 gap-3">
                <Button variant="secondary" onClick={() => { setMessage(null); setCalibrationResult(null); setStage("setup"); }}>Cancel</Button>
                <Button disabled={mode === "upload" && !uploadedFrame} onClick={() => void continueCalibration()}>
                  {mode === "upload" ? (calibrationResult ? "Redetect" : "Detect") : "Continue"}
                </Button>
              </div>
            </Card>
          )}
          {stage === "solving-calibration" && (
            <Card>
              <CalibrationStatus status="Solving" message="Checking stumps..." />
              <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full w-1/2 animate-pulse rounded-full bg-lime" /></div>
            </Card>
          )}
          {stage === "setup-complete" && (
            <Card>
              <CalibrationStatus status="Setup Complete" message="Setup Complete — both stump sets detected." />
              <DetectionSummary result={calibrationResult} />
              <div className="mt-6 space-y-3">
                {mode === "camera" && <Button className="w-full" onClick={() => setStage("capturing")}>Start Capture</Button>}
                <Button className="w-full" variant="secondary" onClick={() => void continueCalibration()}>Redetect</Button>
                {mode === "upload" && (
                  <label htmlFor="calibration-image-complete" className="block cursor-pointer rounded-xl border border-white/15 bg-white/5 px-5 py-3 text-center text-sm font-bold transition hover:bg-white/10">Upload another image</label>
                )}
                <input id="calibration-image-complete" className="sr-only" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => void handleUpload(event)} />
              </div>
            </Card>
          )}
          {stage === "capturing" && mode === "camera" && <DeliveryCapturePanel deliveryCount={deliveryCount} />}
          {stage === "results" && (
            <Card>
              <CalibrationStatus status="Failed" message="No analysed delivery result is available yet." />
              <Button className="mt-6 w-full" variant="secondary" onClick={() => setStage("capturing")}>Return to Capture</Button>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
