"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getExperimentalSession,
  listExperimentalSessions,
  type ExperimentalSession,
  type ExperimentalSessionDelivery
} from "@/lib/api";


const DELIVERY_SLOTS = 6;


function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(value));
}


function shortId(id: string): string {
  return id.length > 20 ? `${id.slice(0, 16)}...` : id;
}


function deliveryCounts(session: ExperimentalSession) {
  return {
    ready: session.deliveries.filter((delivery) => delivery.analysis_status === "ready").length,
    processing: session.deliveries.filter((delivery) => delivery.analysis_status === "queued" || delivery.analysis_status === "processing").length,
    failed: session.deliveries.filter((delivery) => delivery.analysis_status === "failed").length
  };
}


function DeliveryResultCard({ index, delivery }: { index: number; delivery?: ExperimentalSessionDelivery }) {
  if (!delivery) {
    return (
      <Card className="min-h-44 shadow-none">
        <p className="font-black">Delivery {index}</p>
        <p className="mt-10 text-sm text-white/35">Delivery not captured</p>
      </Card>
    );
  }

  const statusLabel = delivery.analysis_status.charAt(0).toUpperCase() + delivery.analysis_status.slice(1);
  return (
    <Card className="overflow-hidden p-0 shadow-none">
      <div className="flex items-center justify-between gap-3 p-4">
        <p className="font-black">Delivery {index}</p>
        <StatusBadge
          label={statusLabel}
          tone={delivery.analysis_status === "ready" ? "good" : delivery.analysis_status === "failed" ? "warn" : "neutral"}
        />
      </div>

      {delivery.analysis_status === "ready" && delivery.processed_video_url && (
        <div className="border-t border-white/10 p-4">
          <video className="aspect-video w-full rounded-xl bg-black object-contain" controls preload="metadata" src={delivery.processed_video_url} />
          <div className="mt-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-lg bg-black/20 p-2"><span className="block text-white/35">Frames</span><strong>{delivery.frames_processed}</strong></div>
            <div className="rounded-lg bg-black/20 p-2"><span className="block text-white/35">With ball</span><strong>{delivery.frames_with_ball}</strong></div>
            <div className="rounded-lg bg-black/20 p-2"><span className="block text-white/35">Best</span><strong>{(delivery.best_confidence * 100).toFixed(1)}%</strong></div>
            <div className="rounded-lg bg-black/20 p-2"><span className="block text-white/35">Average</span><strong>{(delivery.average_confidence * 100).toFixed(1)}%</strong></div>
          </div>
          <a className="mt-3 inline-flex rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-bold hover:bg-white/10" href={delivery.processed_video_url} target="_blank" rel="noreferrer">Open video</a>
        </div>
      )}

      {(delivery.analysis_status === "queued" || delivery.analysis_status === "processing") && (
        <div className="border-t border-white/10 p-4 text-sm text-white/55">
          <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#ffca68]" />
          <span className="ml-2">Processing ball detection{delivery.progress > 0 ? ` · ${delivery.progress}%` : "..."}</span>
        </div>
      )}

      {delivery.analysis_status === "failed" && (
        <div className="border-t border-signal/20 bg-signal/[0.04] p-4">
          <p className="text-sm leading-6 text-[#ffaaa6]">{delivery.error_message ?? "Ball detection failed."}</p>
          {delivery.raw_video_url && <a className="mt-3 inline-flex text-xs font-bold text-white/60 underline" href={delivery.raw_video_url} target="_blank" rel="noreferrer">Open raw clip</a>}
        </div>
      )}
    </Card>
  );
}


export default function SessionResultsPage() {
  const [sessions, setSessions] = useState<ExperimentalSession[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<ExperimentalSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectedIdRef = useRef<string | null>(null);

  const openSession = useCallback(async (sessionId: string, showLoading = true, select = true) => {
    if (select) {
      selectedIdRef.current = sessionId;
      setSelectedId(sessionId);
      setSelectedSession(null);
    }
    if (showLoading) setRefreshing(true);
    try {
      const session = await getExperimentalSession(sessionId);
      if (selectedIdRef.current !== sessionId) return false;
      setSelectedSession(session);
      setSessions((current) => current.map((item) => item.id === session.id ? session : item));
      setError(null);
      return true;
    } catch (caught) {
      if (selectedIdRef.current === sessionId) {
        setError(caught instanceof Error ? caught.message : "Could not load session results.");
        if (select) setSelectedSession(null);
      }
      return false;
    } finally {
      if (showLoading) setRefreshing(false);
    }
  }, []);

  const loadSessions = useCallback(async (showPageLoading = true) => {
    if (showPageLoading) setLoading(true);
    else setRefreshing(true);
    try {
      const records = await listExperimentalSessions();
      setSessions(records);
      const requestedId = new URLSearchParams(window.location.search).get("session_id");
      const currentId = selectedIdRef.current;
      const initialId = currentId && records.some((record) => record.id === currentId)
        ? currentId
        : requestedId && records.some((record) => record.id === requestedId)
          ? requestedId
          : records[0]?.id;
      if (initialId) {
        await openSession(initialId, false);
      }
      else {
        selectedIdRef.current = null;
        setSelectedId(null);
        setSelectedSession(null);
        setError(null);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not load sessions.");
    } finally {
      if (showPageLoading) setLoading(false);
      else setRefreshing(false);
    }
  }, [openSession]);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (!selectedSession) return;
    const hasActiveJobs = selectedSession.deliveries.some((delivery) => delivery.analysis_status === "queued" || delivery.analysis_status === "processing");
    if (!hasActiveJobs && selectedSession.capture_status !== "recording") return;
    const timer = window.setInterval(() => void openSession(selectedSession.id, false, false), 2000);
    return () => window.clearInterval(timer);
  }, [openSession, selectedSession]);

  const refresh = () => loadSessions(false);

  return (
    <div className="mx-auto max-w-7xl py-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <StatusBadge label="Experimental sessions" tone="good" />
          <h1 className="mt-5 text-4xl font-black tracking-tight">Session Results</h1>
          <p className="mt-3 text-white/50">Review captured deliveries and processed ball-detection clips.</p>
        </div>
        <Button variant="secondary" disabled={loading || refreshing} onClick={() => void refresh()}>{refreshing ? "Refreshing..." : "Refresh Results"}</Button>
      </div>

      {error && <div className="mt-6 rounded-xl border border-signal/25 bg-signal/10 p-4 text-sm text-[#ffaaa6]">{error}</div>}
      {loading && <Card className="mt-8"><p className="animate-pulse text-white/50">Loading session results...</p></Card>}
      {!loading && sessions.length === 0 && <Card className="mt-8"><p className="font-bold">No experimental sessions yet.</p><p className="mt-2 text-sm text-white/45">Capture deliveries from Experimental Delivery Test to create a session.</p></Card>}

      {!loading && sessions.length > 0 && (
        <div className="mt-8 grid gap-6 xl:grid-cols-[22rem_1fr]">
          <aside className="space-y-3">
            {sessions.map((session) => {
              const counts = deliveryCounts(session);
              return (
                <Card key={session.id} className={`shadow-none ${selectedId === session.id ? "border-lime/40 bg-lime/[0.04]" : ""}`}>
                  <p className="font-black">{session.name}</p>
                  <p className="mt-1 font-mono text-[11px] text-white/35" title={session.id}>{shortId(session.id)}</p>
                  <p className="mt-4 text-sm text-white/55">{formatDate(session.created_at)}</p>
                  <p className="mt-2 text-sm font-bold">{session.delivery_count} {session.delivery_count === 1 ? "delivery" : "deliveries"}</p>
                  <p className="mt-1 text-xs text-white/40">{counts.ready} ready · {counts.processing} processing{counts.failed ? ` · ${counts.failed} failed` : ""}</p>
                  <p className="mt-1 text-xs capitalize text-white/35">{session.capture_status.replace("_", " ")}</p>
                  <Button className="mt-4 w-full" variant={selectedId === session.id ? "primary" : "secondary"} onClick={() => void openSession(session.id)}>Open Session</Button>
                </Card>
              );
            })}
          </aside>

          <section>
            {selectedSession ? (
              <>
                <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
                  <div><p className="text-xs font-bold uppercase tracking-[0.15em] text-lime">{selectedSession.capture_status.replace("_", " ")}</p><h2 className="mt-1 text-2xl font-black">{selectedSession.name}</h2></div>
                  <p className="font-mono text-xs text-white/35">{selectedSession.id}</p>
                </div>
                <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
                  {Array.from({ length: DELIVERY_SLOTS }, (_, slot) => {
                    const index = slot + 1;
                    return <DeliveryResultCard key={index} index={index} delivery={selectedSession.deliveries.find((item) => item.delivery_index === index)} />;
                  })}
                </div>
              </>
            ) : <Card><p className="text-white/50">Select a session to review its deliveries.</p></Card>}
          </section>
        </div>
      )}
    </div>
  );
}
