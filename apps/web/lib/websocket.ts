import { API_BASE_URL } from "./api";


export function createCricVisionSocket(path = "/live"): WebSocket {
  const url = process.env.NEXT_PUBLIC_WS_URL
    ?? API_BASE_URL.replace(/^http/, "ws");
  return new WebSocket(`${url}${path}`);
}
