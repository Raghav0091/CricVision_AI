import type { BoxLayout, PitchOverlay, PixelPoint, StumpDetection, VirtualStumpEnd, VirtualStumpGeometry } from "@/lib/types";


// Measured from reference footage, normalised to the preview box. Both boxes
// share a horizontal centre of ~0.5015 — keep that symmetry if these are ever
// retuned. UPLOAD_ALIGNMENT_BOXES below is a separate layout for still images
// and is deliberately not derived from these.
export const CAMERA_ALIGNMENT_BOXES: BoxLayout = {
  striker: { x: 0.347, y: 0.339, width: 0.309, height: 0.202 },
  non_striker: { x: 0.199, y: 0.586, width: 0.605, height: 0.265 }
};


export const UPLOAD_ALIGNMENT_BOXES: BoxLayout = {
  striker: { x: 0.455, y: 0.38, width: 0.09, height: 0.18 },
  non_striker: { x: 0.40, y: 0.68, width: 0.20, height: 0.27 }
};


type DetectionMap = Record<"striker" | "non_striker", StumpDetection>;
type Point = { x: number; y: number };

export type MappingLandmarks = { raw: PixelPoint[]; accepted: PixelPoint[] };

const GUIDE_RED = "#E0201F";
const LANDMARK_ACCEPTED_BLUE = "#1A3FD6";

// Regulation wicket: outer stump centres sit +/- 95.25 mm from the middle, so
// the centre-to-centre span of the outer pair is 0.1905 m against a maximum
// stump diameter of 0.0381 m. Drawn thickness is that true ratio against the
// measured pixel span — see cricket_pitch_geometry.py, which is the single
// source of truth for these dimensions. Exaggerating them hides misalignment.
const STUMP_DIAMETER_TO_OUTER_SPAN = 0.0381 / 0.1905;


function AlignmentBox({ label, box }: { label: string; box: { x: number; y: number; width: number; height: number } }) {
  return (
    <div
      className="absolute border-2 border-dashed"
      style={{
        left: `${box.x * 100}%`,
        top: `${box.y * 100}%`,
        width: `${box.width * 100}%`,
        height: `${box.height * 100}%`,
        borderColor: GUIDE_RED
      }}
    >
      <span className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-sm bg-[#D9D9D9] px-2 py-0.5 text-[10px] font-bold text-black">
        {label}
      </span>
    </div>
  );
}


function lerpPoint(a: Point, b: Point, t: number): Point {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}


// ponytail: length zones (yorker/full/good/short) are analysis graphics, not
// setup graphics. During live calibration the operator is checking that the
// wicket sits on the real stumps, and coloured bands with 13px labels bury
// exactly the thing they need to see. They belong on the replay, not here.


// Regulation crease geometry, in metres. Source of truth is
// packages/cricket_vision/calibration/cricket_pitch_geometry.py — these are
// mirrored here only as ratios against the corridor, which is exactly one
// wicket wide (WICKET_WIDTH_M) and one pitch long (PITCH_LENGTH_M).
const PITCH_LENGTH_M = 20.12;
const WICKET_WIDTH_M = 0.2286;
const POPPING_CREASE_OFFSET_M = 1.22;
const RETURN_CREASE_OFFSET_M = 1.32;

// Lateral extent expressed in corridor half-widths: the corridor half-width is
// WICKET_WIDTH_M / 2, so a return crease at 1.32m sits 11.55 half-widths out.
const RETURN_CREASE_K = RETURN_CREASE_OFFSET_M / (WICKET_WIDTH_M / 2);
// Longitudinal offset as a fraction of pitch length.
const CREASE_T = POPPING_CREASE_OFFSET_M / PITCH_LENGTH_M;


// The full crease box at one end: the popping crease in front of the stumps and
// the two return creases running back past them. The bowling crease itself is
// drawn from crease_guides, which already sits on the stump line.
//
// ponytail: positions are interpolated linearly along the corridor rather than
// projected through a camera matrix, so they drift slightly at the far end.
// PitchGrid already makes the same approximation; both become exact for free
// once the overlay is driven by a solved pose instead of stump bboxes.
function CreaseBox({ corridor, end }: { corridor: PitchOverlay["pitch_corridor"]; end: "striker" | "non_striker" }) {
  if (corridor.length !== 4) return null;
  const [nonStrikerLeft, strikerLeft, strikerRight, nonStrikerRight] = corridor;

  // t runs 0 at the striker end to 1 at the non-striker end.
  const edgesAt = (t: number) => ({
    left: lerpPoint(strikerLeft, nonStrikerLeft, t),
    right: lerpPoint(strikerRight, nonStrikerRight, t)
  });
  // k is measured in corridor half-widths either side of the centre line.
  const pointAt = (t: number, k: number): Point => {
    const { left, right } = edgesAt(t);
    const centre = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 };
    return { x: centre.x + k * (right.x - centre.x), y: centre.y + k * (right.y - centre.y) };
  };

  const atStump = end === "striker" ? 0 : 1;
  const inward = end === "striker" ? 1 : -1;
  const popping = atStump + inward * CREASE_T;
  const behind = atStump - inward * CREASE_T;

  const poppingLeft = pointAt(popping, -RETURN_CREASE_K);
  const poppingRight = pointAt(popping, RETURN_CREASE_K);

  return (
    <g stroke="rgba(255,255,255,0.85)" strokeWidth="3" strokeLinecap="round" vectorEffect="non-scaling-stroke">
      <line x1={poppingLeft.x} y1={poppingLeft.y} x2={poppingRight.x} y2={poppingRight.y} />
      {[-RETURN_CREASE_K, RETURN_CREASE_K].map((k) => {
        const front = pointAt(popping, k);
        const back = pointAt(behind, k);
        return <line key={`return-${end}-${k}`} x1={front.x} y1={front.y} x2={back.x} y2={back.y} />;
      })}
    </g>
  );
}


// Faint grid over the mapped pitch surface, signalling the ground plane has
// been solved. Divisions are interpolated across the four corridor corners, so
// the grid inherits the same camera perspective as the corridor itself. Kept
// deliberately dim — it is a confirmation cue, not the subject of the frame.
function PitchGrid({ corridor }: { corridor: PitchOverlay["pitch_corridor"] }) {
  if (corridor.length !== 4) return null;
  const [nonStrikerLeft, strikerLeft, strikerRight, nonStrikerRight] = corridor;
  const LONG_DIVISIONS = 10;
  const LAT_DIVISIONS = 4;

  const longLines = Array.from({ length: LONG_DIVISIONS + 1 }, (_, i) => {
    const t = i / LONG_DIVISIONS;
    return { left: lerpPoint(strikerLeft, nonStrikerLeft, t), right: lerpPoint(strikerRight, nonStrikerRight, t) };
  });
  const latLines = Array.from({ length: LAT_DIVISIONS + 1 }, (_, j) => {
    const t = j / LAT_DIVISIONS;
    return { near: lerpPoint(strikerLeft, strikerRight, t), far: lerpPoint(nonStrikerLeft, nonStrikerRight, t) };
  });

  return (
    <g opacity="0.28">
      {longLines.map((line, i) => (
        <line
          key={`grid-long-${i}`}
          x1={line.left.x}
          y1={line.left.y}
          x2={line.right.x}
          y2={line.right.y}
          stroke="#4ADE80"
          // Fade toward the striker end so the far half does not collapse into
          // a moire band once the lines project closer than a pixel apart.
          strokeOpacity={0.25 + 0.75 * (i / LONG_DIVISIONS)}
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
      ))}
      {latLines.map((line, i) => (
        <line
          key={`grid-lat-${i}`}
          x1={line.near.x}
          y1={line.near.y}
          x2={line.far.x}
          y2={line.far.y}
          stroke="#4ADE80"
          strokeOpacity="0.5"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </g>
  );
}


// One wicket: three gold cylinders plus the two bails bridging them. Each
// cylinder is a 4-point silhouette (top and base can differ in x under
// perspective) capped by an ellipse, which reads as a solid round object
// without needing a 3D renderer.
function VirtualWicket({ end, wicket }: { end: string; wicket: VirtualStumpEnd | null | undefined }) {
  const stumps = wicket?.stumps ?? [];
  if (stumps.length < 3) return null;
  const outerSpan = Math.abs(stumps[2].top.x - stumps[0].top.x) || 1;
  const thickness = Math.max(2, outerSpan * STUMP_DIAMETER_TO_OUTER_SPAN);
  const half = thickness / 2;
  const capRy = Math.max(1, thickness * 0.3);

  return (
    <g>
      {stumps.map((stump) => (
        <g key={`${end}-${stump.name}`}>
          <polygon
            points={[
              `${stump.top.x - half},${stump.top.y}`,
              `${stump.top.x + half},${stump.top.y}`,
              `${stump.base.x + half},${stump.base.y}`,
              `${stump.base.x - half},${stump.base.y}`
            ].join(" ")}
            fill="url(#stump-gold)"
          />
          <ellipse cx={stump.top.x} cy={stump.top.y} rx={half} ry={capRy} fill="#F2C75C" />
        </g>
      ))}
      {(wicket?.bails ?? []).map((bail) => {
        const bailHeight = Math.max(1.5, thickness * 0.55);
        return (
          <polygon
            key={`${end}-${bail.name}`}
            points={[
              `${bail.start.x},${bail.start.y - bailHeight}`,
              `${bail.end.x},${bail.end.y - bailHeight}`,
              `${bail.end.x},${bail.end.y}`,
              `${bail.start.x},${bail.start.y}`
            ].join(" ")}
            fill="url(#stump-gold)"
          />
        );
      })}
    </g>
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
  mappingLandmarks,
  setupComplete = false
}: {
  detections?: DetectionMap | null;
  virtualStumps?: VirtualStumpGeometry | null;
  pitchOverlay?: PitchOverlay | null;
  boxLayout?: BoxLayout;
  frameWidth?: number;
  frameHeight?: number;
  showAlignment?: boolean;
  mappingLandmarks?: MappingLandmarks | null;
  setupComplete?: boolean;
}) {
  const canDrawResults = Boolean(frameWidth && frameHeight && (detections || mappingLandmarks));
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
          // "slice" scales the frame uniformly and center-crops it to fill the
          // container — matching the <video>'s object-cover behavior. "none"
          // (the old value) stretched x/y independently, which is invisible
          // when the container's aspect ratio is close to the frame's but
          // badly squishes/distorts everything (stumps included) on the
          // full-screen live camera view, where the container is much taller
          // than the captured frame.
          preserveAspectRatio="xMidYMid slice"
          aria-label="Detected and estimated stump geometry"
        >
          <defs>
            <filter id="wicket-glow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
            <linearGradient id="stump-gold" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#E8B33A" />
              <stop offset="100%" stopColor="#C98F24" />
            </linearGradient>
          </defs>

          {/* SVG has no z-buffer: far wicket, then the ground corridor, then
              the near wicket, so the near stumps occlude the pitch surface. */}
          <VirtualWicket end="striker" wicket={virtualStumps?.striker} />

          {pitchOverlay?.pitch_corridor.length === 4 && (
            <polygon
              points={pitchOverlay.pitch_corridor.map((point) => `${point.x},${point.y}`).join(" ")}
              fill="#8E93C8"
              fillOpacity="0.42"
            />
          )}
          {pitchOverlay?.pitch_corridor.length === 4 && <PitchGrid corridor={pitchOverlay.pitch_corridor} />}
          {/* ponytail: no centre line. It runs straight down the corridor the
              virtual stumps sit in, so it hides the one alignment cue that
              matters — whether the gold wicket is seated on the real one. */}
          {pitchOverlay && (["striker", "non_striker"] as const).map((end) => {
            const guide = pitchOverlay.crease_guides[end];
            return (
              <g key={`${end}-creases`}>
                {/* Bowling crease — already sits on the stump line. */}
                {guide.length === 2 && (
                  <line
                    x1={guide[0].x}
                    y1={guide[0].y}
                    x2={guide[1].x}
                    y2={guide[1].y}
                    stroke="rgba(255,255,255,0.85)"
                    strokeWidth="3"
                    strokeLinecap="round"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                <CreaseBox corridor={pitchOverlay.pitch_corridor} end={end} />
              </g>
            );
          })}

          <VirtualWicket end="non_striker" wicket={virtualStumps?.non_striker} />

          {/* Mapping stage: raw detection (red) and smoothed/accepted (blue)
              landmarks drawn together so the user watches them converge. */}
          {mappingLandmarks?.raw.map((point, index) => (
            <circle key={`raw-${index}`} cx={point.x} cy={point.y} r="5" fill={GUIDE_RED} />
          ))}
          {mappingLandmarks?.accepted.map((point, index) => (
            <circle key={`accepted-${index}`} cx={point.x} cy={point.y} r="5" fill={LANDMARK_ACCEPTED_BLUE} />
          ))}
        </svg>
      )}
      {setupComplete && virtualStumps && (
        <span className="absolute bottom-3 right-3 rounded bg-ink/80 px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-[#ffe761]">
          Estimated from bounding box
        </span>
      )}
    </div>
  );
}
