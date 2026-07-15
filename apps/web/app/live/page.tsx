"use client";

import { useRef, useState } from "react";

import { CalibrationStatus } from "@/components/live/CalibrationStatus";
import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { DeliveryCapturePanel } from "@/components/live/DeliveryCapturePanel";
import { ALIGNMENT_BOXES, StumpAlignmentOverlay } from "@/components/live/StumpAlignmentOverlay";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { solveCalibration } from "@/lib/api";
import type { CalibrationResponse, LiveStage } from "@/lib/types";


const setupItems = [
  ["Fixed camera", "Use a tripod or stable phone mount."],
  ["Six stumps visible", "Keep both stump sets clearly inside the frame."],
  ["Behind non-striker", "Place the camera behind the non-striker stumps."],
  ["Good lighting", "Avoid shadows, glare, and low-light motion blur."],
  ["Clear view", "Keep players and equipment from blocking the camera."]
];


function failureMessage(result: CalibrationResponse): string {
  if (result.status === "stump_detector_missing") {
    return "Stump detector model is missing. Add a model at Models/stump_detector/best.pt.";
  }
  if (result.status === "stumps_not_found") {
    return "Place real cricket stumps inside both boxes and try again.";
  }
  return result.message;
}


function DetectionSummary({ result }: { result: CalibrationResponse | null }) {
  if (!result?.detections) return null;
  return (
    <div className="mt-5 rounded-xl border border-white/10 bg-white/[0.03] p-4">
      <div className="flex items-center justify-between">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/45">Detector evidence</p>
        {result.virtual_stumps && <span className="text-[10px] font-bold uppercase text-[#ffe761]">Estimated from bounding box</span>}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        {(["striker", "non_striker"] as const).map((end) => {
          const detection = result.detections?.[end];
          return (
            <div key={end} className="rounded-lg bg-black/20 p-3">
              <p className="text-xs capitalize text-white/45">{end}</p>
              <p className={`mt-1 text-sm font-bold ${detection?.found ? "text-lime" : "text-[#ffaaa6]"}`}>
                {detection?.found ? `${(detection.confidence * 100).toFixed(1)}%` : "Not found"}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}


export default function LivePage() {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const [stage, setStage] = useState<LiveStage>("setup");
  const [message, setMessage] = useState<string | null>(null);
  const [calibrationResult, setCalibrationResult] = useState<CalibrationResponse | null>(null);
  const [frameSize, setFrameSize] = useState<{ width: number; height: number } | null>(null);
  const [deliveryCount] = useState(0);

  async function continueCalibration() {
    const frame = cameraRef.current?.captureFrame();
    if (!frame) {
      setMessage("Camera frame is not ready. Wait a moment and try again.");
      return;
    }
    setFrameSize({ width: frame.width, height: frame.height });
    setMessage(null);
    setCalibrationResult(null);
    setStage("solving-calibration");
    try {
      const result = await solveCalibration(frame, ALIGNMENT_BOXES);
      setCalibrationResult(result);
      if (!result.success) {
        setMessage(failureMessage(result));
        setStage("align-stumps");
        return;
      }
      setStage("setup-complete");
    } catch (error) {
      setMessage(error instanceof Error && error.message !== "Failed to fetch" ? error.message : "Calibration backend is unavailable. Confirm FastAPI is running on port 8000.");
      setStage("align-stumps");
    }
  }

  if (stage === "setup") {
    return (
      <div className="mx-auto max-w-6xl py-5">
        <StatusBadge label="Live bowling session" tone="good" />
        <div className="mt-6 grid gap-8 lg:grid-cols-[1.25fr_0.75fr] lg:items-end">
          <div>
            <h1 className="text-5xl font-black tracking-[-0.05em] sm:text-7xl">Set the pitch.<br /><span className="text-lime">Capture the spell.</span></h1>
            <p className="mt-6 max-w-xl text-base leading-7 text-white/55">Start with a fixed camera and an honest stump calibration. CricVision will not show analysis until the setup is validated.</p>
          </div>
          <Button className="w-full py-4 text-base" onClick={() => { setMessage(null); setCalibrationResult(null); setStage("align-stumps"); }}>Start Live Delivery Analysis</Button>
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

  return (
    <div className="mx-auto max-w-7xl py-2">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-lime">Live session</p>
          <h1 className="mt-1 text-2xl font-black">Camera setup</h1>
        </div>
        <StatusBadge label={stage.replaceAll("-", " ")} tone={stage === "setup-complete" || stage === "capturing" ? "good" : "neutral"} />
      </div>
      <div className="grid gap-5 xl:grid-cols-[1fr_22rem]">
        <div className="relative">
          <CameraPreview ref={cameraRef} />
          {(stage === "align-stumps" || stage === "solving-calibration" || stage === "setup-complete" || stage === "capturing") && (
            <StumpAlignmentOverlay
              showAlignment={stage === "align-stumps" || stage === "solving-calibration"}
              detections={calibrationResult?.detections}
              virtualStumps={calibrationResult?.virtual_stumps}
              frameWidth={frameSize?.width}
              frameHeight={frameSize?.height}
            />
          )}
          {stage === "capturing" && <div className="absolute left-4 top-4 rounded-full bg-signal px-3 py-1 text-xs font-black uppercase tracking-wider">Live</div>}
        </div>
        <div className="space-y-4">
          {stage === "align-stumps" && (
            <Card>
              <CalibrationStatus status={message ? "Failed" : "Searching"} message={message ?? "Fit both stump sets inside the red boxes, then press Continue."} />
              <DetectionSummary result={calibrationResult} />
              <div className="mt-6 grid grid-cols-2 gap-3">
                <Button variant="secondary" onClick={() => { setMessage(null); setCalibrationResult(null); setStage("setup"); }}>Cancel</Button>
                <Button onClick={() => void continueCalibration()}>Continue</Button>
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
              <CalibrationStatus status="Setup Complete" message={calibrationResult?.message ?? "Both stump sets detected. Pitch setup is ready."} />
              <DetectionSummary result={calibrationResult} />
              <div className="mt-6 space-y-3">
                <Button className="w-full" onClick={() => setStage("capturing")}>Start Capture</Button>
                <Button className="w-full" variant="secondary" onClick={() => { setMessage(null); setCalibrationResult(null); setStage("align-stumps"); }}>Redetect</Button>
              </div>
            </Card>
          )}
          {stage === "capturing" && <DeliveryCapturePanel deliveryCount={deliveryCount} />}
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
