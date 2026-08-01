"use client";

import { Canvas } from "@react-three/fiber";
import { Component, useEffect, useState, type ErrorInfo, type ReactNode } from "react";

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

  const requestedCap = props.visualOptions.dprCap ?? 2;
  const dprCap = props.visualOptions.lowPerformance
    ? 1
    : Math.max(1, Math.min(2, requestedCap));
  const calibratedMode = props.mode === "camera-validation" || props.mode === "real-frame-overlay";
  const transparent = props.mode === "real-frame-overlay";
  const overlayOpacity = transparent
    ? Math.max(0, Math.min(1, props.visualOptions.overlayOpacity ?? 1))
    : 1;

  if (calibratedMode && !props.calibratedCamera) {
    return <RendererFallback message="A calibrated camera is required for this renderer mode." />;
  }

  return (
    <RendererErrorBoundary>
      <Canvas
        camera={{
          fov: props.camera.verticalFovDegrees,
          near: props.camera.near,
          far: props.camera.far
        }}
        dpr={[1, dprCap]}
        fallback={<RendererFallback message="WebGL is unavailable. Enable hardware acceleration to view the 3D pitch." />}
        flat
        frameloop="demand"
        gl={{
          // Alpha must be enabled at context creation so source switching can
          // enter overlay mode without recreating the whole Canvas.
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
          width: "100%"
        }}
      >
        <VirtualPitchScene {...props} />
      </Canvas>
    </RendererErrorBoundary>
  );
}
