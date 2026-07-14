export function createCricVisionSocket(path = "/live"): WebSocket {
  const url = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
  return new WebSocket(`${url}${path}`);
}
