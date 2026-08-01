import type { LandmarkProjectionComparison } from "./types";


export function ProjectionDiagnosticsOverlay({
  comparisons,
  imageWidth,
  imageHeight,
  showOpenCv,
  showThree,
  showResiduals,
  showLabels
}: {
  comparisons: LandmarkProjectionComparison[];
  imageWidth: number;
  imageHeight: number;
  showOpenCv: boolean;
  showThree: boolean;
  showResiduals: boolean;
  showLabels: boolean;
}) {
  return (
    <svg
      aria-label="OpenCV and Three.js landmark residual diagnostics"
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
    >
      {comparisons.map((point) => {
        const cv = point.opencv_pixel;
        const three = point.three_pixel;
        return (
          <g key={point.semantic_id}>
            {showResiduals && cv && three && (
              <line
                x1={cv.x}
                y1={cv.y}
                x2={three.x}
                y2={three.y}
                stroke="#ff5f87"
                strokeWidth="1.5"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {showOpenCv && cv && (
              <circle
                cx={cv.x}
                cy={cv.y}
                r="5"
                fill="none"
                stroke="#ffe56b"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {showThree && three && (
              <path
                d={`M ${three.x - 5} ${three.y} L ${three.x + 5} ${three.y} M ${three.x} ${three.y - 5} L ${three.x} ${three.y + 5}`}
                fill="none"
                stroke="#5ee7ff"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            )}
            {showLabels && (three ?? cv) && (
              <text
                x={(three ?? cv)!.x + 7}
                y={(three ?? cv)!.y - 7}
                fill="#ffffff"
                fontSize="11"
                paintOrder="stroke"
                stroke="#07110d"
                strokeWidth="3"
              >
                {point.semantic_id}{point.error_px == null ? "" : ` ${point.error_px.toFixed(2)} px`}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
