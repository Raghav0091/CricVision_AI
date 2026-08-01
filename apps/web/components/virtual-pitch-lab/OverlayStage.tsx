"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { containedMediaRect, type ContainedMediaRect } from "@/lib/calibrationCoordinates";
import type { ProjectedPitchGeometry } from "@/lib/api";
import { validRenderBounds } from "@/lib/virtual-pitch/cameraOwnership";

import { ProjectedPitchSvg } from "./ProjectedPitchSvg";
import { ProjectionDiagnosticsOverlay } from "./ProjectionDiagnosticsOverlay";
import type { LandmarkProjectionComparison, OverlayComparisonMode } from "./types";


type StageContentProps = {
  imageWidth: number;
  imageHeight: number;
  frameUrl?: string | null;
  threeCanvas: ReactNode;
  projection?: ProjectedPitchGeometry | null;
  comparisons: LandmarkProjectionComparison[];
  comparisonMode: OverlayComparisonMode;
  overlayOpacity: number;
  showOpenCvMarkers: boolean;
  showThreeMarkers: boolean;
  showResiduals: boolean;
  showLabels: boolean;
};


function StageContent({
  imageWidth,
  imageHeight,
  frameUrl,
  threeCanvas,
  projection,
  comparisons,
  comparisonMode,
  overlayOpacity,
  showOpenCvMarkers,
  showThreeMarkers,
  showResiduals,
  showLabels
}: StageContentProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [mediaRect, setMediaRect] = useState<ContainedMediaRect | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const update = () => {
      const bounds = container.getBoundingClientRect();
      if (validRenderBounds(bounds.width, bounds.height)) {
        setMediaRect(containedMediaRect(
          { width: imageWidth, height: imageHeight },
          { width: bounds.width, height: bounds.height }
        ));
      } else {
        setMediaRect(null);
      }
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(container);
    return () => observer.disconnect();
  }, [imageHeight, imageWidth]);

  const showSvg = comparisonMode === "svg" || comparisonMode === "both";
  const showThree = comparisonMode === "three" || comparisonMode === "both";
  return (
    <div ref={containerRef} className="relative h-full min-h-[22rem] overflow-hidden bg-[#030604]">
      {mediaRect && (
        <div
          className="absolute overflow-hidden bg-black"
          data-native-height={imageHeight}
          data-native-width={imageWidth}
          style={{ left: mediaRect.x, top: mediaRect.y, width: mediaRect.width, height: mediaRect.height }}
        >
          {frameUrl ? (
            // The backend owns this browser-safe setup-frame URL.
            // eslint-disable-next-line @next/next/no-img-element
            <img alt="Selected calibration setup frame" className="absolute inset-0 h-full w-full" draggable={false} src={frameUrl} />
          ) : (
            <div className="absolute inset-0 bg-[#07110d]" />
          )}
          {showSvg && projection && (
            <ProjectedPitchSvg projection={projection} opacity={overlayOpacity} className="pointer-events-none absolute inset-0 h-full w-full" />
          )}
          {showThree && (
            <div className="absolute inset-0" style={{ opacity: overlayOpacity }}>{threeCanvas}</div>
          )}
          <ProjectionDiagnosticsOverlay
            comparisons={comparisons}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
            showOpenCv={showOpenCvMarkers}
            showThree={showThreeMarkers}
            showResiduals={showResiduals}
            showLabels={showLabels}
          />
        </div>
      )}
    </div>
  );
}


export function OverlayStage(props: StageContentProps & { onCanvasCountChange?: (count: number) => void }) {
  const { onCanvasCountChange, ...stageProps } = props;
  const stageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !onCanvasCountChange) return;
    const update = () => onCanvasCountChange(stage.querySelectorAll("canvas").length);
    update();
    const observer = new MutationObserver(update);
    observer.observe(stage, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [onCanvasCountChange, props.comparisonMode]);

  if (props.comparisonMode === "side-by-side") {
    return (
      <div ref={stageRef} className="grid h-full min-h-[22rem] gap-px bg-white/10 lg:grid-cols-2">
        <div className="relative min-h-[22rem]">
          <span className="absolute left-3 top-3 z-20 bg-black/70 px-2 py-1 text-[10px] font-black uppercase text-white">OpenCV SVG</span>
          <StageContent {...stageProps} comparisonMode="svg" />
        </div>
        <div className="relative hidden min-h-[22rem] lg:block">
          <span className="absolute left-3 top-3 z-20 bg-black/70 px-2 py-1 text-[10px] font-black uppercase text-white">Three.js</span>
          <StageContent {...stageProps} comparisonMode="three" />
        </div>
        <p className="grid min-h-[5rem] place-items-center px-4 text-center text-xs text-white/45 lg:hidden">Side-by-side comparison is available on wider screens. Select one overlay on mobile.</p>
      </div>
    );
  }
  return <div ref={stageRef} className="h-full"><StageContent {...stageProps} /></div>;
}
