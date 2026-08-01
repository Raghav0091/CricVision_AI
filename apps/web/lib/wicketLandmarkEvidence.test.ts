// @ts-expect-error Node's type-stripping test runner requires an explicit TypeScript extension.
import { safeAnalysisMediaUrl } from "./wicketLandmarkMedia.ts";


function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}


const API_BASE_URL = "http://127.0.0.1:8000";
const mediaUrl = (value: string | null | undefined) => safeAnalysisMediaUrl(API_BASE_URL, value);

const ownedRelative = mediaUrl(
  "/video-analysis/analysis_example/wicket-landmark-evidence/debug-frame/42"
);
assert(ownedRelative?.startsWith(API_BASE_URL), "Analysis-owned relative media URL was not resolved against the API.");

const apiOrigin = new URL(API_BASE_URL).origin;
const ownedAbsolute = mediaUrl(
  `${apiOrigin}/video-analysis/analysis_example/wicket-landmark-evidence/debug-frame/42`
);
assert(ownedAbsolute !== null, "Same-origin API media URL was rejected.");

const ownedStatic = mediaUrl(
  "/static/video-analysis/analysis_example/calibration/wicket_landmarks_v1/near_temporal_consensus.png"
);
assert(ownedStatic !== null, "Validated analysis-owned static debug media was rejected.");

assert(mediaUrl(null) === null, "Null media must remain unavailable.");
assert(mediaUrl("") === null, "Empty media must remain unavailable.");
assert(mediaUrl("C:\\outputs\\private.png") === null, "Windows filesystem path was exposed.");
assert(mediaUrl("../reports/private.png") === null, "Parent traversal URL was exposed.");
assert(mediaUrl("https://example.com/debug.png") === null, "Cross-origin debug media was exposed.");
assert(mediaUrl("//example.com/debug.png") === null, "Protocol-relative cross-origin media was exposed.");
assert(mediaUrl("/internal/debug.png") === null, "Non-analysis API media was exposed.");
assert(mediaUrl("/static/video-analysis/analysis_example/raw/original_video.mp4") === null, "Non-landmark static media was exposed.");
assert(mediaUrl("/video-analysis/analysis_example/%2e%2e/private.png") === null, "Encoded traversal URL was exposed.");

console.log("Wicket landmark evidence media URL checks passed.");
