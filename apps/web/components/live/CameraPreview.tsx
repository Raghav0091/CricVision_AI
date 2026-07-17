"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

import type { CapturedFrame } from "@/lib/types";


export type CameraPreviewHandle = {
  captureFrame: () => CapturedFrame | null;
  sampleFrame: (width: number, height: number) => ImageData | null;
  getStream: () => MediaStream | null;
};


export const CameraPreview = forwardRef<CameraPreviewHandle>(function CameraPreview(_, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function openCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
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

  useImperativeHandle(ref, () => ({
    captureFrame() {
      const video = videoRef.current;
      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth) return null;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
      return { dataUrl: canvas.toDataURL("image/jpeg", 0.9), width: canvas.width, height: canvas.height };
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
      context.drawImage(video, 0, 0, width, height);
      return context.getImageData(0, 0, width, height);
    },
    getStream() {
      return streamRef.current;
    }
  }));

  return (
    <div className="relative aspect-video overflow-hidden rounded-2xl bg-black">
      <video ref={videoRef} autoPlay muted playsInline className="h-full w-full object-cover" />
      {error && <div className="absolute inset-0 grid place-items-center bg-ink/95 p-8 text-center text-sm text-[#ffaaa6]">{error}</div>}
    </div>
  );
});
