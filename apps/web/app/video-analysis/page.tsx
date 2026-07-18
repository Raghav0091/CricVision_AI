"use client";

import { type ChangeEvent, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getVideoAnalysis,
  prepareVideoAnalysis,
  type VideoAnalysisPreparedResponse
} from "@/lib/api";


type WorkspaceState = "idle" | "file_selected" | "uploading" | "prepared" | "failed";
type ActiveStage = "upload" | "calibration";


const MAX_FILE_BYTES = 500 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = new Set(["mp4", "mov", "webm", "avi", "mkv"]);
const ACCEPT_VALUE = "video/mp4,video/webm,video/quicktime,video/x-msvideo,video/x-matroska";


function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`;
}


function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return minutes ? `${minutes}m ${remainder.toFixed(1)}s` : `${remainder.toFixed(2)}s`;
}


function validateVideo(file: File): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !ACCEPTED_EXTENSIONS.has(extension)) {
    return "Choose an MP4, MOV, WebM, AVI, or MKV video.";
  }
  if (file.size === 0) return "The selected video is empty.";
  if (file.size > MAX_FILE_BYTES) return "The selected video exceeds the 500 MB limit.";
  return null;
}


function WorkflowStage({
  index,
  title,
  state,
  note
}: {
  index: number;
  title: string;
  state: "active" | "complete" | "available" | "locked" | "disabled";
  note: string;
}) {
  const stateStyles = {
    active: "border-lime/40 bg-lime/[0.08]",
    complete: "border-lime/20 bg-lime/[0.03]",
    available: "border-[#ffca68]/30 bg-[#ffca68]/[0.04]",
    locked: "border-white/10 bg-white/[0.02] opacity-65",
    disabled: "border-white/10 bg-white/[0.02] opacity-55"
  };
  return (
    <div className={`rounded-xl border p-4 ${stateStyles[state]}`}>
      <div className="flex items-center justify-between gap-3">
        <span className={`text-xs font-black ${state === "disabled" ? "text-white/30" : "text-lime"}`}>0{index}</span>
        <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
          {state === "disabled" ? "Coming later" : note}
        </span>
      </div>
      <p className="mt-5 text-sm font-black">{title}</p>
    </div>
  );
}


function MetadataGrid({ analysis }: { analysis: VideoAnalysisPreparedResponse }) {
  const values = [
    ["File size", formatBytes(analysis.file_size_bytes)],
    ["Duration", formatDuration(analysis.duration_seconds)],
    ["FPS", analysis.fps.toFixed(3).replace(/\.?0+$/, "")],
    ["Resolution", `${analysis.width} × ${analysis.height}`],
    ["Total frames", analysis.frame_count.toLocaleString()],
    ["Codec", analysis.codec ?? "Unavailable"],
    ["Reference frame", analysis.reference_frame_index.toLocaleString()]
  ];
  return (
    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
      {values.map(([label, value]) => (
        <div key={label} className="rounded-xl bg-black/20 p-3">
          <span className="block text-xs text-white/35">{label}</span>
          <strong className="mt-1 block text-sm">{value}</strong>
        </div>
      ))}
    </div>
  );
}


export default function VideoAnalysisPage() {
  const previewUrlRef = useRef<string | null>(null);
  const [workspaceState, setWorkspaceState] = useState<WorkspaceState>("idle");
  const [activeStage, setActiveStage] = useState<ActiveStage>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<VideoAnalysisPreparedResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(false);

  function releasePreview() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
  }

  function clearPreparedUrl() {
    window.history.replaceState(null, "", "/video-analysis");
  }

  function selectFile(file: File) {
    releasePreview();
    const objectUrl = URL.createObjectURL(file);
    previewUrlRef.current = objectUrl;
    setPreviewUrl(objectUrl);
    setSelectedFile(file);
    setAnalysis(null);
    setActiveStage("upload");
    setWorkspaceState("file_selected");
    setError(null);
    clearPreparedUrl();
  }

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const validationError = validateVideo(file);
    if (validationError) {
      setError(validationError);
      setWorkspaceState(selectedFile ? "file_selected" : "failed");
      return;
    }
    selectFile(file);
  }

  function removeSelectedFile() {
    releasePreview();
    setSelectedFile(null);
    setAnalysis(null);
    setError(null);
    setActiveStage("upload");
    setWorkspaceState("idle");
    clearPreparedUrl();
  }

  async function uploadAndPrepare() {
    if (!selectedFile) {
      setError("Select a video before preparing the analysis.");
      setWorkspaceState("failed");
      return;
    }
    const validationError = validateVideo(selectedFile);
    if (validationError) {
      setError(validationError);
      setWorkspaceState("failed");
      return;
    }

    setWorkspaceState("uploading");
    setError(null);
    try {
      const prepared = await prepareVideoAnalysis(selectedFile);
      setAnalysis(prepared);
      setWorkspaceState("prepared");
      setActiveStage("upload");
      window.history.replaceState(
        null,
        "",
        `/video-analysis?analysis_id=${encodeURIComponent(prepared.analysis_id)}`
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Video preparation failed.");
      setWorkspaceState("failed");
    }
  }

  useEffect(() => {
    const analysisId = new URLSearchParams(window.location.search).get("analysis_id");
    if (!analysisId) return;
    let cancelled = false;
    setRestoring(true);
    setWorkspaceState("uploading");
    void getVideoAnalysis(analysisId)
      .then((restored) => {
        if (cancelled) return;
        setAnalysis(restored);
        setWorkspaceState("prepared");
        setError(null);
      })
      .catch((caught) => {
        if (cancelled) return;
        setError(caught instanceof Error ? caught.message : "The saved analysis could not be restored.");
        setWorkspaceState("failed");
      })
      .finally(() => {
        if (!cancelled) setRestoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  const uploadComplete = workspaceState === "prepared" && analysis !== null;
  const calibrationActive = uploadComplete && activeStage === "calibration";

  return (
    <div className="mx-auto max-w-7xl py-5">
      <StatusBadge label="Video Analysis" tone="good" />
      <div className="mt-5 max-w-3xl">
        <h1 className="text-4xl font-black tracking-tight sm:text-5xl">Upload and prepare a cricket video.</h1>
        <p className="mt-4 leading-7 text-white/50">Create a persistent analysis workspace, inspect the source metadata, and extract a clean calibration reference frame.</p>
      </div>

      <div className="mt-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <WorkflowStage index={1} title="Upload Video" state={uploadComplete ? "complete" : "active"} note={uploadComplete ? "Completed" : "Active"} />
        <WorkflowStage index={2} title="Scene Calibration" state={calibrationActive ? "active" : uploadComplete ? "available" : "locked"} note={calibrationActive ? "Active" : uploadComplete ? "Available" : "Locked"} />
        <WorkflowStage index={3} title="Ball Detection" state="disabled" note="" />
        <WorkflowStage index={4} title="Ball Tracking" state="disabled" note="" />
        <WorkflowStage index={5} title="Physics and Replay" state="disabled" note="" />
      </div>

      <Card className="mt-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-xl font-black">Upload Video</h2>
            <p className="mt-2 text-sm text-white/45">MP4, MOV, WebM, AVI, or MKV · maximum 500 MB</p>
          </div>
          <StatusBadge
            label={workspaceState === "uploading" ? "Preparing" : uploadComplete ? "Prepared" : selectedFile ? "File selected" : "Waiting"}
            tone={workspaceState === "failed" ? "warn" : uploadComplete ? "good" : "neutral"}
          />
        </div>

        {error && <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm leading-6 text-[#ffaaa6]">{error}</p>}

        {!selectedFile && !uploadComplete && !restoring && (
          <label htmlFor="analysis-video" className="mt-6 block cursor-pointer rounded-2xl border border-dashed border-white/20 bg-white/[0.03] px-6 py-12 text-center transition hover:border-lime/40 hover:bg-lime/[0.03]">
            <span className="block text-lg font-black">Choose one cricket video</span>
            <span className="mt-2 block text-sm text-white/40">The file stays local until you press Upload and Prepare Analysis.</span>
          </label>
        )}
        <input id="analysis-video" className="sr-only" type="file" accept={ACCEPT_VALUE} onChange={handleFileSelection} />

        {selectedFile && (
          <div className="mt-6 grid gap-5 lg:grid-cols-[1fr_19rem]">
            <video className="aspect-video w-full rounded-xl bg-black object-contain" controls preload="metadata" src={previewUrl ?? undefined} />
            <div className="rounded-xl border border-white/10 bg-black/20 p-4">
              <p className="break-all font-bold">{selectedFile.name}</p>
              <p className="mt-2 text-sm text-white/45">{formatBytes(selectedFile.size)}</p>
              <div className="mt-6 space-y-2">
                <label htmlFor="analysis-video" className="block cursor-pointer rounded-xl border border-white/15 bg-white/5 px-4 py-3 text-center text-sm font-bold hover:bg-white/10">Change selected file</label>
                <Button className="w-full" variant="danger" disabled={workspaceState === "uploading"} onClick={removeSelectedFile}>Remove selected file</Button>
              </div>
            </div>
          </div>
        )}

        {workspaceState === "uploading" && (
          <div className="mt-6 rounded-xl border border-lime/20 bg-lime/[0.04] p-4">
            <p className="font-bold text-lime">{restoring ? "Restoring prepared analysis..." : "Uploading and preparing video..."}</p>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10"><div className="h-full w-1/2 animate-pulse rounded-full bg-lime" /></div>
          </div>
        )}

        {selectedFile && workspaceState !== "prepared" && (
          <Button className="mt-6 w-full sm:w-auto" disabled={workspaceState === "uploading"} onClick={() => void uploadAndPrepare()}>
            {workspaceState === "uploading" ? "Uploading and preparing..." : "Upload and Prepare Analysis"}
          </Button>
        )}
      </Card>

      {uploadComplete && analysis && (
        <section className="mt-6 space-y-5">
          <Card>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <StatusBadge label="Prepared" tone="good" />
                <h2 className="mt-4 text-2xl font-black">Video prepared</h2>
                <p className="mt-2 break-all font-mono text-xs text-white/40">{analysis.analysis_id}</p>
                <p className="mt-2 text-sm text-white/55">{analysis.original_filename}</p>
              </div>
              <label htmlFor="analysis-video" className="cursor-pointer rounded-xl border border-white/15 bg-white/5 px-4 py-3 text-sm font-bold hover:bg-white/10">Prepare another video</label>
            </div>
            <div className="mt-6"><MetadataGrid analysis={analysis} /></div>
          </Card>

          <div className="grid gap-5 xl:grid-cols-2">
            <Card>
              <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">Original video</p>
              <video className="mt-4 aspect-video w-full rounded-xl bg-black object-contain" controls preload="metadata" src={analysis.original_video_url} />
              <a className="mt-3 inline-flex text-xs font-bold text-lime underline" href={analysis.original_video_url} target="_blank" rel="noreferrer">Open original video</a>
            </Card>
            <Card>
              <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">Calibration reference frame</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="mt-4 h-auto w-full rounded-xl bg-black object-contain" src={analysis.reference_frame_url} alt={`Calibration reference frame ${analysis.reference_frame_index}`} />
              <p className="mt-3 text-xs text-white/45">Middle frame · index {analysis.reference_frame_index}</p>
            </Card>
          </div>

          {!calibrationActive && (
            <Button onClick={() => setActiveStage("calibration")}>Continue to Scene Calibration</Button>
          )}

          {calibrationActive && (
            <Card className="border-[#ffca68]/25">
              <StatusBadge label="Scene Calibration" />
              <h2 className="mt-4 text-2xl font-black">Calibration reference frame</h2>
              <p className="mt-3 text-sm leading-6 text-white/55">Stump detection and manual wicket confirmation will be added in the next milestone.</p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className="mt-5 h-auto w-full rounded-xl bg-black object-contain" src={analysis.reference_frame_url} alt="Clean scene calibration reference" />
            </Card>
          )}
        </section>
      )}
    </div>
  );
}
