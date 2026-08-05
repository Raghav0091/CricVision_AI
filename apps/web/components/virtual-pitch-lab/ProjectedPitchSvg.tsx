import type { PitchProjectionGeometry } from "@/lib/api";


function lineStyle(category: string) {
  if (category === "centreline") return { colour: "#c7ff7a", dash: "8 7", width: 2 };
  if (category === "return_crease") return { colour: "#79d8ff", width: 2 };
  if (category === "pitch_boundary") return { colour: "#6df0a5", width: 2 };
  return { colour: "#ffffff", width: 2.5 };
}


export function ProjectedPitchSvg({
  projection,
  opacity = 1,
  className = ""
}: {
  projection: PitchProjectionGeometry;
  opacity?: number;
  className?: string;
}) {
  const { source_camera: camera } = projection;
  return (
    <svg
      aria-label="Backend OpenCV pitch projection"
      className={className}
      viewBox={`0 0 ${camera.image_width} ${camera.image_height}`}
      preserveAspectRatio="none"
    >
      <g opacity={opacity}>
        {projection.projected_polygons.map((polygon) => {
          if (!polygon.projection_valid || polygon.pixel_vertices.some((point) => !point)) return null;
          const points = polygon.pixel_vertices.map((point) => `${point?.x},${point?.y}`).join(" ");
          const corridor = polygon.polygon_category.includes("corridor");
          return (
            <polygon
              key={polygon.primitive_id}
              points={points}
              fill={corridor ? "#70ffad" : "#1f8f55"}
              fillOpacity={corridor ? 0.14 : 0.22}
              stroke={corridor ? "#b5ffd3" : "#63e99c"}
              strokeDasharray={corridor ? "7 6" : undefined}
              strokeWidth="2"
            />
          );
        })}
        {projection.projected_line_segments.map((line) => {
          if (!line.projection_valid || !line.pixel_start || !line.pixel_end) return null;
          const style = lineStyle(line.line_category);
          return (
            <line
              key={line.primitive_id}
              x1={line.pixel_start.x}
              y1={line.pixel_start.y}
              x2={line.pixel_end.x}
              y2={line.pixel_end.y}
              stroke={style.colour}
              strokeDasharray={style.dash}
              strokeLinecap="round"
              strokeWidth={style.width}
            />
          );
        })}
        {projection.projected_stumps.map((stump) => {
          if (!stump.projection_valid || !stump.pixel_base || !stump.pixel_top) return null;
          return (
            <line
              key={stump.primitive_id}
              x1={stump.pixel_base.x}
              y1={stump.pixel_base.y}
              x2={stump.pixel_top.x}
              y2={stump.pixel_top.y}
              stroke="#ffeaa4"
              strokeLinecap="round"
              strokeWidth={Math.max(2, (stump.projected_radius_px ?? 1) * 2)}
            />
          );
        })}
        {projection.projected_bails.map((bail) => bail.projection_valid && bail.pixel_start && bail.pixel_end ? (
          <line
            key={bail.primitive_id}
            x1={bail.pixel_start.x}
            y1={bail.pixel_start.y}
            x2={bail.pixel_end.x}
            y2={bail.pixel_end.y}
            stroke="#ffd45f"
            strokeLinecap="round"
            strokeWidth="2.5"
          />
        ) : null)}
      </g>
    </svg>
  );
}
