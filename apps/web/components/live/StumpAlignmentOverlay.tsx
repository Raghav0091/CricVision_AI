import type { BoxLayout, PitchOverlay, StumpDetection, VirtualStumpGeometry } from "@/lib/types";


export const CAMERA_ALIGNMENT_BOXES: BoxLayout = {
  striker: { x: 0.43, y: 0.22, width: 0.14, height: 0.24 },
  non_striker: { x: 0.35, y: 0.66, width: 0.30, height: 0.28 }
};


export const UPLOAD_ALIGNMENT_BOXES: BoxLayout = {
  striker: { x: 0.455, y: 0.38, width: 0.09, height: 0.18 },
  non_striker: { x: 0.40, y: 0.68, width: 0.20, height: 0.27 }
};


type DetectionMap = Record<"striker" | "non_striker", StumpDetection>;


function AlignmentBox({ label, box }: { label: string; box: { x: number; y: number; width: number; height: number } }) {
  return (
    <div
      className="absolute rounded-md border-2 border-dashed border-signal shadow-[0_0_18px_rgba(255,85,79,0.32)]"
      style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }}
    >
      <span className="absolute -top-7 left-0 whitespace-nowrap rounded bg-signal px-2 py-1 text-[10px] font-black uppercase tracking-wide text-white">{label}</span>
    </div>
  );
}


export function StumpAlignmentOverlay({
  detections,
  virtualStumps,
  pitchOverlay,
  boxLayout = CAMERA_ALIGNMENT_BOXES,
  frameWidth,
  frameHeight,
  showAlignment = true,
  setupComplete = false
}: {
  detections?: DetectionMap | null;
  virtualStumps?: VirtualStumpGeometry | null;
  pitchOverlay?: PitchOverlay | null;
  boxLayout?: BoxLayout;
  frameWidth?: number;
  frameHeight?: number;
  showAlignment?: boolean;
  setupComplete?: boolean;
}) {
  const canDrawResults = Boolean(frameWidth && frameHeight && detections);
  return (
    <div className="pointer-events-none absolute inset-0">
      {showAlignment && (
        <>
          <AlignmentBox label="Striker Stumps" box={boxLayout.striker} />
          <AlignmentBox label="Non-Striker Stumps" box={boxLayout.non_striker} />
        </>
      )}
      {canDrawResults && (
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${frameWidth} ${frameHeight}`}
          preserveAspectRatio="none"
          aria-label="Detected and estimated stump geometry"
        >
          <defs>
            <filter id="wicket-glow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          {pitchOverlay?.pitch_corridor.length === 4 && (
            <polygon
              points={pitchOverlay.pitch_corridor.map((point) => `${point.x},${point.y}`).join(" ")}
              fill="rgba(226, 183, 72, 0.16)"
              stroke="rgba(255, 225, 132, 0.75)"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {pitchOverlay?.center_line.length === 2 && (
            <line
              x1={pitchOverlay.center_line[0].x}
              y1={pitchOverlay.center_line[0].y}
              x2={pitchOverlay.center_line[1].x}
              y2={pitchOverlay.center_line[1].y}
              stroke="rgba(255,255,255,0.8)"
              strokeWidth="2"
              strokeDasharray="8 8"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {pitchOverlay && (["striker", "non_striker"] as const).map((end) => {
            const guide = pitchOverlay.crease_guides[end];
            if (guide.length !== 2) return null;
            return <line key={`${end}-crease`} x1={guide[0].x} y1={guide[0].y} x2={guide[1].x} y2={guide[1].y} stroke="rgba(255,255,255,0.72)" strokeWidth="2" vectorEffect="non-scaling-stroke" />;
          })}
          {(["striker", "non_striker"] as const).map((end) => {
            const detection = detections?.[end];
            const bbox = detection?.bbox;
            if (!detection?.found || !bbox) return null;
            return (
              <g key={`${end}-detection`}>
                <rect
                  x={bbox.x}
                  y={bbox.y}
                  width={bbox.width}
                  height={bbox.height}
                  fill="rgba(183,243,75,0.08)"
                  stroke="#b7f34b"
                  strokeWidth="2"
                  rx="5"
                  strokeDasharray="7 5"
                  vectorEffect="non-scaling-stroke"
                />
                <text x={bbox.x + 5} y={Math.max(16, bbox.y - 7)} fill="#b7f34b" fontSize="14" fontWeight="700">
                  {`${end} ${(detection.confidence * 100).toFixed(0)}%`}
                </text>
              </g>
            );
          })}
          {virtualStumps && (["striker", "non_striker"] as const).flatMap((end) =>
            (virtualStumps[end]?.stumps ?? []).map((stump) => (
              <line
                key={`${end}-${stump.name}`}
                x1={stump.top.x}
                y1={stump.top.y}
                x2={stump.base.x}
                y2={stump.base.y}
                stroke="#f6cf62"
                strokeWidth="8"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
                filter="url(#wicket-glow)"
              />
            ))
          )}
          {virtualStumps && (["striker", "non_striker"] as const).flatMap((end) =>
            (virtualStumps[end]?.bails ?? []).map((bail) => (
              <line
                key={`${end}-${bail.name}`}
                x1={bail.start.x}
                y1={bail.start.y}
                x2={bail.end.x}
                y2={bail.end.y}
                stroke="#ffdf7e"
                strokeWidth="7"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
                filter="url(#wicket-glow)"
              />
            ))
          )}
        </svg>
      )}
      {virtualStumps && (
        <span className="absolute bottom-3 right-3 rounded bg-ink/80 px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-[#ffe761]">
          Estimated from bounding box
        </span>
      )}
      {setupComplete && (
        <span className="absolute left-3 top-3 rounded-full border border-lime/40 bg-ink/85 px-3 py-1.5 text-[11px] font-black uppercase tracking-wide text-lime shadow-lg">
          Setup Complete
        </span>
      )}
    </div>
  );
}
