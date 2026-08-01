"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  AnalysisFailureState,
  DeliveryMetrics,
  DeveloperDiagnostics,
  PitchFitStatus,
  PitchSpaceEntry,
  PitchSpaceProgress,
  PitchSpaceTimeline,
  RealVideoReplay,
  VirtualPitchReplay
} from "@/components/pitch-space-analysis";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getPitchModel,
  getPitchSpaceAnalysis,
  getRecentPitchSpaceAnalyses,
  pitchSpaceVideoUrl,
  runPitchSpaceAnalysis,
  uploadPitchSpaceVideo
} from "@/lib/pitch-space-analysis/api";
import { frameAtTime, timeAtFrame } from "@/lib/pitch-space-analysis/replay";
import type { PitchSpaceAnalysis, RecentAnalysis } from "@/lib/pitch-space-analysis/types";
import type { VirtualPitchModel } from "@/lib/virtual-pitch/types";

const MAX_FILE_BYTES = 500 * 1024 * 1024;

export default function PitchSpaceAnalysisPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [analysis, setAnalysis] = useState<PitchSpaceAnalysis | null>(null);
  const [pitch, setPitch] = useState<VirtualPitchModel | null>(null);
  const [recent, setRecent] = useState<RecentAnalysis[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [fullTrail, setFullTrail] = useState(false);

  const present = useCallback((result: PitchSpaceAnalysis, model: VirtualPitchModel) => {
    setAnalysis(result);
    setPitch(model);
    setFrame(0);
    setPlaying(false);
    setError(null);
    window.history.replaceState(null, "", `/pitch-space-analysis?analysis_id=${encodeURIComponent(result.analysis_id)}`);
  }, []);

  const load = useCallback(async (analysisId: string) => {
    setBusy(true);
    setProgress(1);
    setError(null);
    try {
      const [result, model] = await Promise.all([
        getPitchSpaceAnalysis(analysisId).catch(() => runPitchSpaceAnalysis(analysisId)),
        getPitchModel()
      ]);
      present(result, model);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The analysis could not be loaded.");
    } finally {
      setBusy(false);
    }
  }, [present]);

  async function upload(file: File) {
    if (file.size === 0 || file.size > MAX_FILE_BYTES) {
      setError(file.size === 0 ? "The selected video is empty." : "The selected video exceeds 500 MB.");
      return;
    }
    setBusy(true);
    setProgress(0);
    setError(null);
    try {
      const [result, model] = await Promise.all([uploadPitchSpaceVideo(file), getPitchModel()]);
      present(result, model);
      setRecent(await getRecentPitchSpaceAnalyses());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Video analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void getRecentPitchSpaceAnalyses().then(setRecent);
    const id = new URLSearchParams(window.location.search).get("analysis_id");
    if (id) void load(id);
  }, [load]);

  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => setProgress((value) => Math.min(10, value + 1)), 1250);
    return () => window.clearInterval(timer);
  }, [busy]);

  function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.playbackRate = speed;
      void video.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      video.pause();
      setPlaying(false);
    }
  }

  function scrub(nextFrame: number) {
    setFrame(nextFrame);
    if (videoRef.current && analysis?.fps) videoRef.current.currentTime = timeAtFrame(nextFrame, analysis.fps);
  }

  function changeSpeed(value: number) {
    setSpeed(value);
    if (videoRef.current) videoRef.current.playbackRate = value;
  }

  const trackCount = analysis?.pitch_space_track?.length ?? 0;
  return (
    <div className="mx-auto max-w-[1560px] space-y-5">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="text-xs font-bold uppercase text-lime">Development lab</p>
          <h1 className="mt-1 text-2xl font-semibold sm:text-3xl">Pitch-Space Delivery Analysis</h1>
        </div>
        <StatusBadge label={busy ? "Processing" : analysis ? analysis.status.replaceAll("_", " ") : "Ready"} tone={error ? "warn" : analysis ? "good" : "neutral"} />
      </header>

      <PitchSpaceEntry busy={busy} recent={recent} onUpload={(file) => void upload(file)} onLoad={(id) => void load(id)} />
      {busy && <PitchSpaceProgress activeIndex={progress} />}
      {error && <div role="alert" className="border-l-2 border-signal bg-signal/10 px-4 py-3 text-sm text-[#ffaaa6]">{error}</div>}

      {analysis && pitch && !busy && (
        <>
          <PitchFitStatus analysis={analysis} />
          <AnalysisFailureState analysis={analysis} />
          <div className="grid gap-5 xl:grid-cols-2">
            <RealVideoReplay
              ref={videoRef}
              analysis={analysis}
              videoUrl={pitchSpaceVideoUrl(analysis.analysis_id)}
              currentFrame={frame}
              fullTrail={fullTrail}
              onTime={(seconds) => setFrame(frameAtTime(seconds, analysis.fps ?? 0, analysis.frame_count ?? 0))}
              onEnded={() => setPlaying(false)}
            />
            <VirtualPitchReplay analysis={analysis} pitch={pitch} currentFrame={frame} fullTrail={fullTrail} />
          </div>
          <PitchSpaceTimeline
            frame={frame}
            frameCount={analysis.frame_count ?? 0}
            fps={analysis.fps ?? 0}
            playing={playing}
            speed={speed}
            fullTrail={fullTrail}
            onPlay={togglePlayback}
            onFrame={scrub}
            onSpeed={changeSpeed}
            onTrail={setFullTrail}
          />
          <div className="flex flex-wrap gap-3 text-xs text-white/45">
            <span>{trackCount} pitch-space points</span>
            <span>Observed: solid lime</span>
            <span>Recovered: dashed blue</span>
            <span>Projected: dotted amber</span>
          </div>
          <DeliveryMetrics analysis={analysis} />
          <DeveloperDiagnostics analysis={analysis} />
        </>
      )}
    </div>
  );
}
