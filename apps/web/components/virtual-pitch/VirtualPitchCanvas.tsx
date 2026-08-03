"use client";

import { Canvas } from "@react-three/fiber";
import { Component, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ErrorInfo, type ReactNode } from "react";

import {
  buildThreeCameraFromOpenCv,
  type CameraBridgeInput,
  type ThreeCameraBridge
} from "@/lib/virtual-pitch/opencvCameraBridge";

import { useOwnedPitchCamera } from "./VirtualPitchCameraController";
import { VirtualPitchScene } from "./VirtualPitchScene";
import type { VirtualPitchSceneProps } from "./rendererTypes";


const fallbackStyle = {
  alignItems: "center",
  display: "flex",
  height: "100%",
  justifyContent: "center",
  minHeight: "280px",
  padding: "24px",
  textAlign: "center" as const,
  width: "100%"
};


function RendererFallback({ message }: { message: string }) {
  return (
    <div role="status" style={fallbackStyle}>
      <p>{message}</p>
    </div>
  );
}


class RendererErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Virtual Pitch renderer failed", error, info.componentStack);
  }

  render() {
    if (this.state.failed) {
      return <RendererFallback message="The 3D pitch could not be rendered in this browser." />;
    }
    return this.props.children;
  }
}


function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}


export function VirtualPitchCanvas(props: VirtualPitchSceneProps) {
  const [webGLAvailable, setWebGLAvailable] = useState<boolean | null>(null);
  useEffect(() => setWebGLAvailable(supportsWebGL()), []);

  if (webGLAvailable === null) {
    return <RendererFallback message="Preparing 3D pitch..." />;
  }
  if (!webGLAvailable) {
    return <RendererFallback message="WebGL is unavailable. Enable hardware acceleration to view the 3D pitch." />;
  }

  const calibratedMode = props.mode === "camera-validation" || props.mode === "real-frame-overlay";

  if (calibratedMode && !props.calibratedCamera) {
    return <RendererFallback message="A calibrated camera is required for this renderer mode." />;
  }

  return <ReadyVirtualPitchCanvas {...props} />;
}


function ReadyVirtualPitchCanvas(props: VirtualPitchSceneProps) {
  const calibratedMode = props.mode === "camera-validation" || props.mode === "real-frame-overlay";
  const bridge = useMemo<ThreeCameraBridge | null>(() => {
    if (!calibratedMode || !props.calibratedCamera) return null;
    return "projectionMatrix" in props.calibratedCamera
      ? props.calibratedCamera
      : buildThreeCameraFromOpenCv(props.calibratedCamera as CameraBridgeInput);
  }, [calibratedMode, props.calibratedCamera]);
  const ownedCamera = useOwnedPitchCamera(props.camera, bridge);
  const [readyCameraUuid, setReadyCameraUuid] = useState<string | null>(null);
  const cameraReady = readyCameraUuid === ownedCamera.camera.uuid;
  const requestedCap = props.visualOptions.dprCap ?? 2;
  const dprCap = props.visualOptions.lowPerformance
    ? 1
    : Math.max(1, Math.min(2, requestedCap));
  const transparent = props.mode === "real-frame-overlay";
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const updateSize = () => {
      const bounds = container.getBoundingClientRect();
      setDisplaySize({ width: bounds.width, height: bounds.height });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);
  const nativeDpr = calibratedMode && bridge && displaySize.width > 0 && displaySize.height > 0
    ? Math.max(
        bridge.intrinsics.imageWidth / displaySize.width,
        bridge.intrinsics.imageHeight / displaySize.height
      ) + 1e-6
    : null;
  const renderDpr: number | [number, number] = nativeDpr ?? [1, dprCap];
  const overlayOpacity = transparent
    ? Math.max(0, Math.min(1, props.visualOptions.overlayOpacity ?? 1))
    : 1;
  const handleReadyChange = useCallback((cameraUuid: string, ready: boolean) => {
    setReadyCameraUuid(ready ? cameraUuid : null);
  }, []);

  return (
    <RendererErrorBoundary>
      <div
        ref={containerRef}
        className="relative h-full w-full"
        data-camera-family={ownedCamera.family}
        data-camera-ready={cameraReady}
        data-native-image-height={bridge?.intrinsics.imageHeight}
        data-native-image-width={bridge?.intrinsics.imageWidth}
      >
        <Canvas
          key={ownedCamera.camera.uuid}
          camera={ownedCamera.camera}
          dpr={renderDpr}
          fallback={<RendererFallback message="WebGL is unavailable. Enable hardware acceleration to view the 3D pitch." />}
          flat
          frameloop="demand"
          gl={{
            alpha: true,
            antialias: !props.visualOptions.lowPerformance,
            powerPreference: props.visualOptions.lowPerformance ? "low-power" : "high-performance"
          }}
          onCreated={({ gl }) => gl.setClearColor(0x000000, 0)}
          shadows={false}
          style={{
            background: transparent ? "transparent" : undefined,
            height: "100%",
            opacity: overlayOpacity,
            pointerEvents: transparent ? "none" : undefined,
            width: "100%"
          }}
        >
          <VirtualPitchScene
            {...props}
            ownedCamera={ownedCamera}
            onCameraReadyChange={handleReadyChange}
          />
        </Canvas>
        {!cameraReady && calibratedMode ? (
          <div className="pointer-events-none absolute inset-0 grid place-items-center bg-black/70 text-sm font-semibold text-white/70">
            Preparing calibrated camera...
          </div>
        ) : null}
      </div>
    </RendererErrorBoundary>
  );
}
