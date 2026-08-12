"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import type { CapturedFrame } from "@/lib/types";


export type CameraPreviewHandle = {
  captureFrame: () => CapturedFrame | null;
  sampleFrame: (width: number, height: number) => ImageData | null;
  getStream: () => MediaStream | null;
  /** >1 when the frame is being cropped in software rather than by the lens. */
  getDigitalZoom: () => number;
  /**
   * Convert a point normalised against the on-screen preview box into one
   * normalised against the actual video frame.
   *
   * The two are not the same. The preview is `object-cover`, so a tall 9:20
   * frame shown in a squarer box has roughly half its height cropped away, and
   * digital zoom crops further. Overlay hit-testing works in preview
   * coordinates; anything sent to the solver has to be in frame coordinates,
   * or every placed stump point lands hundreds of pixels from where the
   * operator put it.
   */
  frameFromPreviewPoint: (point: { x: number; y: number }) => { x: number; y: number } | null;
};


const MAX_DIGITAL_ZOOM = 4;

// Rear cameras other than the main sensor advertise themselves in the label.
// Scoring beats matching because label text varies by vendor and locale, and
// an unlabelled list should still resolve to something sensible.
const NON_MAIN_HINTS = ["ultra", "wide angle", "0.5", "telephoto", "tele", "depth", "macro", "mono", "infrared"];

export function pickMainRearCamera(devices: MediaDeviceInfo[]): string | undefined {
  if (devices.length === 0) return undefined;
  const rear = devices.filter((device) => {
    const label = device.label.toLowerCase();
    return label.includes("back") || label.includes("rear") || label.includes("environment");
  });
  const pool = rear.length > 0 ? rear : devices;
  const scored = pool.map((device) => {
    const label = device.label.toLowerCase();
    let score = 0;
    for (const hint of NON_MAIN_HINTS) if (label.includes(hint)) score -= 10;
    // Android commonly names the primary sensor "camera2 0" / "back camera 0".
    if (/\b0\b/.test(label)) score += 3;
    if (label.includes("main")) score += 5;
    return { device, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored[0]?.device.deviceId;
}


export const CameraPreview = forwardRef<CameraPreviewHandle, { className?: string }>(function CameraPreview({ className }, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  // What the browser actually granted, not what we asked for. Surfaced so a
  // poor result can be blamed on capture before anyone debugs the pipeline.
  const [captureInfo, setCaptureInfo] = useState<{ width: number; height: number; frameRate: number } | null>(null);

  // Optical zoom is applied to the track, so the captured frame already
  // reflects it. Digital zoom only scales the <video> in CSS, so captureFrame
  // has to crop to match — otherwise the user aligns against a zoomed preview
  // and calibration receives the un-zoomed frame.
  const [digitalZoom, setDigitalZoom] = useState(1);
  const digitalZoomRef = useRef(1);
  const pointersRef = useRef(new Map<number, { x: number; y: number }>());
  const pinchStartRef = useRef<{ distance: number; optical: number; digital: number } | null>(null);
  const opticalRef = useRef<{ min: number; max: number; step: number; value: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function openCamera() {
      try {
        // A phone exposes several rear cameras — main, ultrawide, depth,
        // sometimes a macro. `facingMode: environment` lets the browser pick
        // any of them, and it often does not pick the main sensor. Open a
        // throwaway stream first so enumerateDevices() returns labels, then
        // choose deliberately.
        let deviceId: string | undefined;
        try {
          const probe = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } } });
          probe.getTracks().forEach((track) => track.stop());
          const devices = await navigator.mediaDevices.enumerateDevices();
          deviceId = pickMainRearCamera(devices.filter((device) => device.kind === "videoinput"));
        } catch {
          // Label-less enumeration or a denied probe just means we fall back
          // to facingMode; not worth failing the whole open for.
        }

        // Ask for the most the sensor will give. The browser negotiates down
        // to the nearest supported mode, so 4K/60 degrades to 1080p/30 on
        // devices that cannot do it rather than throwing.
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: "environment" } }),
            width: { ideal: 3840 },
            height: { ideal: 2160 },
            frameRate: { ideal: 60, min: 30 }
          }
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;

        const track = stream.getVideoTracks()[0];
        const negotiated = track?.getSettings?.();
        if (negotiated) {
          setCaptureInfo({
            width: negotiated.width ?? 0,
            height: negotiated.height ?? 0,
            frameRate: Math.round(negotiated.frameRate ?? 0)
          });
        }
        // `zoom` is not in the standard MediaTrackCapabilities type but is
        // widely implemented on mobile; absence is the documented fallback.
        const capabilities = track?.getCapabilities?.() as { zoom?: { min: number; max: number; step: number } } | undefined;
        if (capabilities?.zoom) {
          const settings = track.getSettings() as { zoom?: number };
          opticalRef.current = {
            min: capabilities.zoom.min,
            max: capabilities.zoom.max,
            step: capabilities.zoom.step || 0.1,
            value: settings.zoom ?? capabilities.zoom.min
          };
        }
      } catch {
        setError("Camera access failed. Check browser permission and use HTTPS or localhost.");
      }
    }
    void openCamera();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  function applyOpticalZoom(value: number) {
    const optical = opticalRef.current;
    const track = streamRef.current?.getVideoTracks()[0];
    if (!optical || !track) return;
    const clamped = Math.max(optical.min, Math.min(optical.max, value));
    optical.value = clamped;
    void track.applyConstraints({ advanced: [{ zoom: clamped }] } as unknown as MediaTrackConstraints).catch(() => {
      // A track that advertises zoom but rejects the constraint should not
      // break the alignment stage; the preview simply stops zooming.
    });
  }

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    pointersRef.current.set(event.pointerId, { x: event.clientX, y: event.clientY });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const pointers = pointersRef.current;
    if (!pointers.has(event.pointerId)) return;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size !== 2) return;

    const [a, b] = Array.from(pointers.values());
    const distance = Math.hypot(a.x - b.x, a.y - b.y);
    if (!pinchStartRef.current) {
      pinchStartRef.current = { distance, optical: opticalRef.current?.value ?? 1, digital: digitalZoomRef.current };
      return;
    }

    const ratio = distance / (pinchStartRef.current.distance || 1);
    if (opticalRef.current) {
      applyOpticalZoom(pinchStartRef.current.optical * ratio);
      return;
    }
    const next = Math.max(1, Math.min(MAX_DIGITAL_ZOOM, pinchStartRef.current.digital * ratio));
    digitalZoomRef.current = next;
    setDigitalZoom(next);
  }

  function onPointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) pinchStartRef.current = null;
  }

  useImperativeHandle(ref, () => ({
    captureFrame() {
      const video = videoRef.current;
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return null;
      // Stump landmark precision sets the ceiling on PnP, and PnP sets the
      // ceiling on every metric — so the calibration frame is the one frame
      // worth spending bytes on. 1920 roughly quadruples the pixels on the far
      // wicket versus the old 1280 cap while staying inside what a Cloudflare
      // quick tunnel will carry. Quality stays at 0.62 to hold the size down.
      const maxWidth = 1920;
      const zoom = digitalZoomRef.current;
      const cropWidth = video.videoWidth / zoom;
      const cropHeight = video.videoHeight / zoom;
      const cropX = (video.videoWidth - cropWidth) / 2;
      const cropY = (video.videoHeight - cropHeight) / 2;

      let width = Math.round(cropWidth);
      let height = Math.round(cropHeight);
      if (width > maxWidth) {
        height = Math.round(height * (maxWidth / width));
        width = maxWidth;
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d")?.drawImage(video, cropX, cropY, cropWidth, cropHeight, 0, 0, width, height);
      return { dataUrl: canvas.toDataURL("image/jpeg", 0.62), width, height };
    },
    sampleFrame(width, height) {
      const video = videoRef.current;
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return null;
      const canvas = sampleCanvasRef.current ?? document.createElement("canvas");
      sampleCanvasRef.current = canvas;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) return null;
      const zoom = digitalZoomRef.current;
      const cropWidth = video.videoWidth / zoom;
      const cropHeight = video.videoHeight / zoom;
      context.drawImage(
        video,
        (video.videoWidth - cropWidth) / 2,
        (video.videoHeight - cropHeight) / 2,
        cropWidth,
        cropHeight,
        0,
        0,
        width,
        height
      );
      return context.getImageData(0, 0, width, height);
    },
    getStream() {
      return streamRef.current;
    },
    getDigitalZoom() {
      return digitalZoomRef.current;
    },
    frameFromPreviewPoint(point) {
      const video = videoRef.current;
      if (!video || !video.videoWidth || !video.videoHeight) return null;
      const box = video.getBoundingClientRect();
      if (box.width <= 0 || box.height <= 0) return null;

      // Undo the CSS scale first: it magnifies about the centre, so the visible
      // window shrinks by the zoom factor.
      const zoom = digitalZoomRef.current || 1;
      const zx = 0.5 + (point.x - 0.5) / zoom;
      const zy = 0.5 + (point.y - 0.5) / zoom;

      // Then undo object-cover: the frame is scaled to cover the box and
      // centre-cropped, so the box shows only the middle of the larger axis.
      const scale = Math.max(box.width / video.videoWidth, box.height / video.videoHeight);
      const shownWidth = video.videoWidth * scale;
      const shownHeight = video.videoHeight * scale;
      const cropX = (shownWidth - box.width) / 2;
      const cropY = (shownHeight - box.height) / 2;

      return {
        x: (zx * box.width + cropX) / shownWidth,
        y: (zy * box.height + cropY) / shownHeight
      };
    }
  }));

  return (
    <div
      className={className ?? "relative aspect-video overflow-hidden rounded-2xl bg-black"}
      style={{ touchAction: "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="h-full w-full object-cover"
        style={digitalZoom > 1 ? { transform: `scale(${digitalZoom})` } : undefined}
      />
      {captureInfo && captureInfo.width > 0 && (
        <div className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-black/60 px-2 py-1 font-mono text-[10px] tabular-nums text-white/70">
          {captureInfo.width}×{captureInfo.height}
          {captureInfo.frameRate > 0 && ` · ${captureInfo.frameRate}fps`}
          {` · ${((captureInfo.width * captureInfo.height) / 1_000_000).toFixed(1)}MP`}
        </div>
      )}
      {error && <div className="absolute inset-0 grid place-items-center bg-ink/95 p-8 text-center text-sm text-[#ffaaa6]">{error}</div>}
    </div>
  );
});
