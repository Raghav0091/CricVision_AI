import type { BoxLayout, StumpDetection, VirtualStumpGeometry } from "@/lib/types";


export const ALIGNMENT_BOXES: BoxLayout = {
  striker: { x: 0.43, y: 0.22, width: 0.14, height: 0.24 },
  non_striker: { x: 0.35, y: 0.66, width: 0.30, height: 0.28 }
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
  frameWidth,
  frameHeight,
  showAlignment = true
}: {
  detections?: DetectionMap | null;
  virtualStumps?: VirtualStumpGeometry | null;
  frameWidth?: number;
  frameHeight?: number;
  showAlignment?: boolean;
}) {
  const canDrawResults = Boolean(frameWidth && frameHeight && detections);
  return (
    <div className="pointer-events-none absolute inset-0">
      {showAlignment && (
        <>
          <AlignmentBox label="Striker Stumps" box={ALIGNMENT_BOXES.striker} />
          <AlignmentBox label="Non-Striker Stumps" box={ALIGNMENT_BOXES.non_striker} />
        </>
      )}
      {canDrawResults && (
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${frameWidth} ${frameHeight}`}
          preserveAspectRatio="none"
          aria-label="Detected and estimated stump geometry"
        >
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
                  strokeWidth="3"
                  vectorEffect="non-scaling-stroke"
                />
                <text x={bbox.x + 5} y={Math.max(16, bbox.y - 7)} fill="#b7f34b" fontSize="14" fontWeight="700">
                  {`${end} ${(detection.confidence * 100).toFixed(0)}%`}
                </text>
              </g>
            );
          })}
          {virtualStumps && (["striker", "non_striker"] as const).flatMap((end) =>
            virtualStumps[end].map((stump) => (
              <line
                key={`${end}-${stump.name}`}
                x1={stump.top.x}
                y1={stump.top.y}
                x2={stump.base.x}
                y2={stump.base.y}
                stroke="#ffe761"
                strokeWidth="3"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
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
    </div>
  );
}
