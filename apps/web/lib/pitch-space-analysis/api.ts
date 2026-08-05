import { adaptVirtualPitchResponse } from "@/lib/virtual-pitch/geometryAdapter";
import type { PitchSpaceAnalysis, RecentAnalysis } from "./types";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL
  ?? process.env.NEXT_PUBLIC_API_URL
  ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function apiError(response: Response, fallback: string): Promise<Error> {
  try {
    const body = await response.json() as { detail?: string; message?: string };
    return new Error(body.detail ?? body.message ?? fallback);
  } catch {
    return new Error(fallback);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store", ...init });
  if (!response.ok) throw await apiError(response, `Pitch-space service returned ${response.status}.`);
  return response.json() as Promise<T>;
}

export async function uploadPitchSpaceVideo(file: File): Promise<PitchSpaceAnalysis> {
  const form = new FormData();
  form.append("video", file);
  return request<PitchSpaceAnalysis>("/pitch-space-analysis/upload", { method: "POST", body: form });
}

export function runPitchSpaceAnalysis(analysisId: string): Promise<PitchSpaceAnalysis> {
  return request(`/pitch-space-analysis/${encodeURIComponent(analysisId)}/run`, { method: "POST" });
}

export function getPitchSpaceAnalysis(analysisId: string): Promise<PitchSpaceAnalysis> {
  return request(`/pitch-space-analysis/${encodeURIComponent(analysisId)}`);
}

export function getRecentPitchSpaceAnalyses(): Promise<RecentAnalysis[]> {
  return request<RecentAnalysis[] | { items: RecentAnalysis[] }>("/pitch-space-analysis/recent")
    .then((value) => Array.isArray(value) ? value : value.items)
    .catch(() => []);
}

export async function getPitchModel() {
  const response = await request<unknown>("/video-analysis/virtual-pitch");
  return adaptVirtualPitchResponse(response);
}

export function pitchSpaceVideoUrl(analysisId: string): string {
  return `${API_BASE_URL}/pitch-space-analysis/${encodeURIComponent(analysisId)}/video`;
}
