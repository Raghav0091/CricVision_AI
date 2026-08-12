/** Stable per-browser camera identity for lens calibration.
 *
 * `MediaDeviceInfo.deviceId` looks like the obvious key and is not one: it is
 * rotated per origin and re-randomised between sessions, so a profile stored
 * against it would be orphaned on the next visit. A UUID we mint ourselves is
 * stable for as long as the browser keeps its storage, which is the same
 * lifetime any calibration can honestly claim.
 */

const DEVICE_ID_KEY = "cricvision.deviceId";
const DEVICE_LABEL_KEY = "cricvision.deviceLabel";


function storage(): Storage | null {
  // Server rendering has no localStorage, and neither does a browser with
  // storage blocked. Both should degrade to "uncalibrated", not throw.
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}


function createDeviceId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // ponytail: only reached on browsers without randomUUID; uniqueness across
  // one user's own phones is all this identifier ever needs.
  return `dev-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}


// Holds the id steady when localStorage is unavailable, so a blocked-storage
// browser still sends one identity for the whole session instead of a new one
// per call.
let cachedDeviceId: string | null = null;


export function getDeviceId(): string {
  const store = storage();
  const existing = store?.getItem(DEVICE_ID_KEY);
  if (existing) {
    cachedDeviceId = existing;
    return existing;
  }
  const created = cachedDeviceId ?? createDeviceId();
  cachedDeviceId = created;
  store?.setItem(DEVICE_ID_KEY, created);
  return created;
}


export function getDeviceLabel(): string | null {
  return storage()?.getItem(DEVICE_LABEL_KEY) || null;
}


export function setDeviceLabel(label: string): void {
  const trimmed = label.trim();
  if (trimmed) storage()?.setItem(DEVICE_LABEL_KEY, trimmed);
  else storage()?.removeItem(DEVICE_LABEL_KEY);
}
