"use client";

import { type ChangeEvent, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import type { RecentAnalysis } from "@/lib/pitch-space-analysis/types";

const ACCEPTED = "video/mp4,video/webm,video/quicktime,video/x-msvideo,video/x-matroska";

export function PitchSpaceEntry({ busy, recent, onUpload, onLoad }: {
  busy: boolean;
  recent: RecentAnalysis[];
  onUpload: (file: File) => void;
  onLoad: (analysisId: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [analysisId, setAnalysisId] = useState("");

  function select(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) onUpload(file);
  }

  return (
    <section className="border-y border-white/10 bg-black/20 py-5">
      <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr] lg:items-end">
        <div>
          <p className="mb-2 text-xs font-bold uppercase text-white/45">New delivery</p>
          <input ref={inputRef} className="sr-only" type="file" accept={ACCEPTED} onChange={select} />
          <Button className="w-full sm:w-auto" disabled={busy} onClick={() => inputRef.current?.click()}>
            Upload video
          </Button>
        </div>
        <span className="hidden pb-3 text-xs font-bold uppercase text-white/25 lg:block">or</span>
        <form onSubmit={(event) => { event.preventDefault(); if (analysisId.trim()) onLoad(analysisId.trim()); }}>
          <label className="mb-2 block text-xs font-bold uppercase text-white/45" htmlFor="analysis-id">Existing analysis</label>
          <div className="flex gap-2">
            <input
              id="analysis-id"
              value={analysisId}
              onChange={(event) => setAnalysisId(event.target.value)}
              placeholder="analysis_..."
              list="recent-pitch-analyses"
              className="min-w-0 flex-1 rounded-md border border-white/15 bg-black/35 px-3 py-2.5 text-sm text-white outline-none focus:border-lime/60"
            />
            <Button variant="secondary" disabled={busy || !analysisId.trim()} type="submit">Load</Button>
          </div>
          <datalist id="recent-pitch-analyses">
            {recent.map((item) => <option key={item.analysis_id} value={item.analysis_id}>{item.original_filename ?? item.source_filename}</option>)}
          </datalist>
        </form>
      </div>
    </section>
  );
}
