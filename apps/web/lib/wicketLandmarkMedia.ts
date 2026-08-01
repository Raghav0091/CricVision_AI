export function safeAnalysisMediaUrl(
  apiBaseUrl: string,
  value: string | null | undefined
): string | null {
  if (!value || value.includes("\\")) return null;
  try {
    const decodedValue = decodeURIComponent(value);
    if (decodedValue.split("/").includes("..") || decodedValue.includes("\\")) return null;
    const api = new URL(apiBaseUrl);
    const resolved = new URL(value, `${apiBaseUrl.replace(/\/$/, "")}/`);
    const decodedPath = decodeURIComponent(resolved.pathname);
    const analysisApiMedia = decodedPath.startsWith("/video-analysis/");
    const analysisStaticMedia = /^\/static\/video-analysis\/analysis_[A-Za-z0-9_-]+\/calibration\/wicket_landmarks_v1\/[A-Za-z0-9_.-]+$/.test(decodedPath);
    if (
      resolved.origin !== api.origin
      || !["http:", "https:"].includes(resolved.protocol)
      || (!analysisApiMedia && !analysisStaticMedia)
      || decodedPath.split("/").includes("..")
    ) return null;
    return resolved.toString();
  } catch {
    return null;
  }
}
