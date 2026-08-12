import { getApiBaseUrl } from "./api";


export function createCricVisionSocket(path = "/live"): WebSocket {
  const url = process.env.NEXT_PUBLIC_WS_URL
    ?? getApiBaseUrl().replace(/^http/, "ws");
  return new WebSocket(`${url}${path}`);
}
