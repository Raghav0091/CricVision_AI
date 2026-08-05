"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Component,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
  type ErrorInfo,
  type ReactNode
} from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import type { VirtualPitchSceneProps } from "@/components/virtual-pitch";
import { getReplayPayload, getVirtualPitchSpecification } from "@/lib/api";
import {
  adaptVirtualPitchResponse,
  calculateCameraPreset,
  materialPreset,
  type VirtualPitchModel
} from "@/lib/virtual-pitch";
import { buildThreeCameraFromOpenCv } from "@/lib/virtual-pitch/opencvCameraBridge";
import {
  EXPORT_HEIGHT,
  EXPORT_WIDTH,
  replayCameraToBridgeInput,
  runDeterministicPlayback,
  useCanvasRecorder,
  useReplayClock,
  validateReplayPayload,
  type ReplayPayloadV1
} from "@/lib/virtual-pitch-replay";

import { ReplayDegradedBanner } from "./ReplayDegradedBanner";
import { ReplayMetricsOverlay } from "./ReplayMetricsOverlay";
import { ReplayPlaybackControls } from "./ReplayPlaybackControls";
import { ReplayTrajectoryLayer } from "./ReplayTrajectoryLayer";

type ReplayCanvasProps = VirtualPitchSceneProps & {
  calibratedCamera?: ReturnType<typeof buildThreeCameraFromOpenCv>;
};

const VirtualPitchCanvas = dynamic<ReplayCanvasProps>(
  () => import("@/components/virtual-pitch").then((module) => module.VirtualPitchCanvas as ComponentType<ReplayCanvasProps>),
  { ssr: false, loading: () => <StageMessage title="Loading 3D replay" detail="Preparing virtual pitch renderer..." /> }
);

function StageMessage({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="grid h-full min-h-[20rem] place-items-center bg-[#050806] p-8 text-center">
      <div className="max-w-sm">
        <p className="text-base font-bold text-white">{title}</p>
        <p className="mt-2 text-sm leading-6 text-white/45">{detail}</p>
      </div>
    </div>
  );
}

class ReplayRendererBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {}

  render() {
    return this.state.error
      ? <StageMessage title="Virtual replay unavailable" detail={this.state.error.message || "The 3D replay could not start."} />
      : this.props.children;
  }
}

export function VirtualPitchReplayStage({ analysisId }: { analysisId: string }) {
  const [payload, setPayload] = useState<ReplayPayloadV1 | null>(null);
  const [model, setModel] = useState<VirtualPitchModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const invalidateRef = useRef<(() => void) | null>(null);
  const recorder = useCanvasRecorder();

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    Promise.all([
      getReplayPayload(analysisId),
      getVirtualPitchSpecification().then(adaptVirtualPitchResponse)
    ])
      .then(([replayPayload, pitchModel]) => {
        if (!active) return;
        setPayload(replayPayload);
        setModel(pitchModel);
      })
      .catch((caught) => {
        if (!active) return;
        setError(caught instanceof Error ? caught.message : "Virtual replay payload is unavailable.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [analysisId]);

  const validation = useMemo(
    () => (payload ? validateReplayPayload(payload) : null),
    [payload]
  );

  const bridgeInput = useMemo(
    () => (payload ? replayCameraToBridgeInput(payload.camera, payload.analysis_id) : null),
    [payload]
  );

  const threeBridge = useMemo(() => {
    if (!bridgeInput) return null;
    try {
      return buildThreeCameraFromOpenCv(bridgeInput);
    } catch {
      return null;
    }
  }, [bridgeInput]);

  const cameraPreset = useMemo(() => {
    if (!model) return null;
    return calculateCameraPreset("bowler-end", model, 16 / 9);
  }, [model]);

  const clock = useReplayClock(payload?.trajectory ?? [], payload?.playback ?? {
    export_width: EXPORT_WIDTH,
    export_height: EXPORT_HEIGHT,
    landscape: true
  }, playing || recorder.isRecording, setPlaying);

  const handleCanvasReady = useCallback((canvas: HTMLCanvasElement, invalidate: () => void) => {
    canvasRef.current = canvas;
    invalidateRef.current = invalidate;
  }, []);

  const handleExport = useCallback(async () => {
    if (!canvasRef.current || !payload || recorder.isRecording) return;
    clock.scrub(clock.windowStartSeconds);

    const blob = await recorder.startRecording({
      canvas: canvasRef.current,
      durationSeconds: Math.max(clock.durationSeconds, 0.1),
      drivePlayback: async ({ onFrame }) => {
        await runDeterministicPlayback({
          startSeconds: clock.windowStartSeconds,
          endSeconds: clock.windowEndSeconds,
          onTime: (timeSeconds, progressRatio) => {
            clock.scrub(timeSeconds);
            invalidateRef.current?.();
            onFrame(progressRatio);
          }
        });
      }
    });

    if (blob) {
      recorder.download(`virtual-pitch-replay-${payload.analysis_id}.webm`);
    }
  }, [clock, payload, recorder]);

  if (loading) {
    return <StageMessage title="Loading virtual replay" detail="Fetching replay payload and pitch geometry..." />;
  }

  if (error || !payload || !model || !cameraPreset) {
    return (
      <StageMessage
        title="Virtual replay unavailable"
        detail={error ?? "Replay payload or pitch geometry is missing."}
      />
    );
  }

  const useCalibratedCamera = Boolean(threeBridge?.renderable);
  const visualOptions = {
    showPitch: true,
    showStumps: true,
    showBails: true,
    showLines: true,
    showCorridor: true,
    enableOrbitControls: false,
    corridorOpacity: 0.24,
    lowPerformance: false,
    materialPreset: materialPreset("cricvision-dark")
  } satisfies VirtualPitchSceneProps["visualOptions"];

  const exportStatus = recorder.error
    ?? (recorder.status === "recording" ? "Recording WebM export"
      : recorder.status === "finalizing" ? "Finalizing export"
        : recorder.status === "complete" ? "Export complete"
          : undefined);

  return (
    <div className="space-y-4">
      <ReplayDegradedBanner
        payload={payload}
        schemaIssues={(validation?.issues ?? []).map((issue) => issue.message)}
        analysisId={analysisId}
      />

      <div className="relative overflow-hidden rounded-xl border border-white/10 bg-[#050806] aspect-video">
        <ReplayRendererBoundary>
          <VirtualPitchCanvas
            model={model}
            mode="interactive-replay"
            camera={cameraPreset}
            calibratedCamera={useCalibratedCamera ? threeBridge ?? undefined : undefined}
            visualOptions={visualOptions}
            frameloop={playing || recorder.isRecording ? "always" : "demand"}
            exportWidth={EXPORT_WIDTH}
            exportHeight={EXPORT_HEIGHT}
            onCanvasReady={handleCanvasReady}
            replayContent={(
              <ReplayTrajectoryLayer
                payload={payload}
                currentSampleIndex={clock.currentSampleIndex}
              />
            )}
          />
        </ReplayRendererBoundary>

        <ReplayMetricsOverlay
          measurementValidity={payload.measurement_validity}
          releaseSpeed={payload.metrics.release_speed_kmh}
          averagePreBounceSpeed={payload.metrics.average_pre_bounce_speed_kmh}
          speedAtBounce={payload.metrics.speed_at_bounce_kmh}
          deliveryLength={payload.metrics.delivery_length_m}
          estimatedLateralDeviation={payload.metrics.estimated_lateral_deviation_m}
        />
      </div>

      <ReplayPlaybackControls
        isPlaying={playing}
        currentTimeSeconds={clock.currentTimeSeconds}
        durationSeconds={clock.durationSeconds}
        windowStartSeconds={clock.windowStartSeconds}
        disabled={payload.trajectory.length === 0}
        exporting={recorder.isRecording}
        onToggle={clock.toggle}
        onRestart={clock.restart}
        onScrub={clock.scrub}
        onExport={handleExport}
        exportStatus={exportStatus}
        exportProgress={recorder.progressRatio}
      />

      <Card className="shadow-none">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/40">Analysis</p>
            <p className="mt-1 text-sm font-bold text-white">{payload.analysis_id}</p>
          </div>
          <Link href={`/video-analysis?analysis_id=${encodeURIComponent(analysisId)}`}>
            <Button variant="secondary" className="px-3 py-2 text-xs">Back to analysis</Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}
