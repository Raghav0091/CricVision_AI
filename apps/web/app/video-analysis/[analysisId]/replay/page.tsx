"use client";

import { useParams } from "next/navigation";

import { VirtualPitchReplayStage } from "@/components/virtual-pitch-replay";

export default function VirtualPitchReplayPage() {
  const params = useParams<{ analysisId: string }>();
  const analysisId = params.analysisId?.trim();

  if (!analysisId) {
    return (
      <div className="mx-auto max-w-5xl py-6">
        <p className="text-sm text-[#ffaaa6]">Analysis ID is required for virtual replay.</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl py-4">
      <header className="mb-4 border-b border-white/10 pb-4">
        <p className="text-xs font-bold uppercase tracking-[0.18em] text-lime">Virtual Pitch Replay</p>
        <h1 className="mt-2 text-2xl font-black sm:text-3xl">3D Delivery Replay</h1>
        <p className="mt-1.5 text-sm text-white/45">
          Pitch-only replay with trajectory animation and exportable WebM.
        </p>
      </header>
      <VirtualPitchReplayStage analysisId={analysisId} />
    </div>
  );
}
