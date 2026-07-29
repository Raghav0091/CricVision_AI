"use client";

import { useEffect, useState } from "react";

import {
  getSyntheticPitchPreview,
  getVirtualPitchSpecification,
  type SyntheticPitchPreviewResponse,
  type VirtualPitchSpecification
} from "@/lib/api";


function cameraLabel(name: string): string {
  return name
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}


function lineStyle(category: string): {
  colour: string;
  dash?: string;
  width: number;
} {
  if (category === "centreline") {
    return { colour: "#b8f276", dash: "8 7", width: 2 };
  }
  if (category === "return_crease") {
    return { colour: "#d7e6dd", width: 2 };
  }
  if (category === "pitch_boundary") {
    return { colour: "#83a890", width: 2 };
  }
  return { colour: "#f2f5ee", width: 2.5 };
}


function ProjectedPitchSvg({
  preview,
  opacity
}: {
  preview: SyntheticPitchPreviewResponse;
  opacity: number;
}) {
  const { projection } = preview;
  const camera = projection.source_camera;
  const corridor = projection.projected_polygons.find(
    (polygon) => polygon.primitive_id === "lbw_stump_to_stump_corridor"
  );
  const surface = projection.projected_polygons.find(
    (polygon) => polygon.primitive_id === "pitch_surface"
  );
  const polygonPoints = (
    polygon: typeof surface
  ): string | null => {
    if (!polygon?.projection_valid || polygon.pixel_vertices.some((point) => !point)) {
      return null;
    }
    return polygon.pixel_vertices
      .map((point) => `${point?.x},${point?.y}`)
      .join(" ");
  };
  const surfacePoints = polygonPoints(surface);
  const corridorPoints = polygonPoints(corridor);

  return (
    <svg
      aria-label="Synthetic virtual pitch projection"
      className="block h-full w-full"
      viewBox={`0 0 ${camera.image_width} ${camera.image_height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <rect
        width={camera.image_width}
        height={camera.image_height}
        fill="#07110d"
      />
      <g opacity={opacity}>
        {surfacePoints && (
          <polygon
            points={surfacePoints}
            fill="#28543a"
            fillOpacity="0.48"
            stroke="#6c9278"
            strokeWidth="2"
          />
        )}
        {corridorPoints && (
          <polygon
            points={corridorPoints}
            fill="#7fe5b0"
            fillOpacity="0.17"
            stroke="#83efb9"
            strokeDasharray="7 6"
            strokeWidth="2"
          />
        )}
        {projection.projected_line_segments.map((line) => {
          if (!line.projection_valid || !line.pixel_start || !line.pixel_end) {
            return null;
          }
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
          if (!stump.projection_valid || !stump.pixel_base || !stump.pixel_top) {
            return null;
          }
          return (
            <line
              key={stump.primitive_id}
              x1={stump.pixel_base.x}
              y1={stump.pixel_base.y}
              x2={stump.pixel_top.x}
              y2={stump.pixel_top.y}
              stroke="#fff6cf"
              strokeLinecap="round"
              strokeWidth={Math.max(2, (stump.projected_radius_px ?? 1) * 2)}
            />
          );
        })}
        {projection.projected_bails.map((bail) => {
          if (!bail.projection_valid || !bail.pixel_start || !bail.pixel_end) {
            return null;
          }
          return (
            <line
              key={bail.primitive_id}
              x1={bail.pixel_start.x}
              y1={bail.pixel_start.y}
              x2={bail.pixel_end.x}
              y2={bail.pixel_end.y}
              stroke="#ffd970"
              strokeLinecap="round"
              strokeWidth="2.5"
            />
          );
        })}
        {projection.projected_landmarks
          .filter((landmark) => (
            landmark.in_frame
            && landmark.pixel_point
            && landmark.semantic_id.endsWith("_middle_stump_base")
          ))
          .map((landmark) => (
            <circle
              key={landmark.semantic_id}
              cx={landmark.pixel_point?.x}
              cy={landmark.pixel_point?.y}
              data-native-x={landmark.pixel_point?.x}
              data-native-y={landmark.pixel_point?.y}
              fill="#ff9c5b"
              r="4"
              stroke="#fff"
              strokeWidth="1.5"
            />
          ))}
      </g>
    </svg>
  );
}


export function VirtualPitchOverlay() {
  const [specification, setSpecification] =
    useState<VirtualPitchSpecification | null>(null);
  const [cameraName, setCameraName] = useState("");
  const [preview, setPreview] =
    useState<SyntheticPitchPreviewResponse | null>(null);
  const [visible, setVisible] = useState(true);
  const [opacity, setOpacity] = useState(0.82);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getVirtualPitchSpecification()
      .then((result) => {
        if (cancelled) return;
        setSpecification(result);
        setCameraName(result.synthetic_camera_names[0] ?? "");
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(
            caught instanceof Error
              ? caught.message
              : "Virtual pitch specification is unavailable."
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!cameraName) return;
    let cancelled = false;
    setError(null);
    void getSyntheticPitchPreview(cameraName)
      .then((result) => {
        if (!cancelled) setPreview(result);
      })
      .catch((caught) => {
        if (!cancelled) {
          setError(
            caught instanceof Error
              ? caught.message
              : "Synthetic projection is unavailable."
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [cameraName]);

  return (
    <section className="border-t border-white/10 pt-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-black">Virtual Pitch Geometry Preview</h3>
          <p className="mt-1 text-xs leading-5 text-[#ffdc9a]">
            Synthetic camera projection. Not registered to this video yet.
          </p>
        </div>
        <span className="border border-white/15 bg-white/5 px-2 py-1 text-[10px] font-bold uppercase text-white/55">
          Model {specification?.virtual_pitch_model_version ?? "v1"}
        </span>
      </div>

      <div className="mt-3 grid min-w-0 gap-3 md:grid-cols-[minmax(0,1fr)_15rem]">
        <div
          className="relative w-full min-w-0 overflow-hidden border border-white/10 bg-[#07110d]"
          style={{
            aspectRatio: preview
              ? `${preview.projection.source_camera.image_width} / ${preview.projection.source_camera.image_height}`
              : "16 / 9"
          }}
        >
          {visible && preview && (
            <ProjectedPitchSvg preview={preview} opacity={opacity} />
          )}
          {!preview && !error && (
            <div className="absolute inset-0 grid place-items-center text-sm text-white/35">
              Loading metric pitch geometry...
            </div>
          )}
          {error && (
            <div className="absolute inset-0 grid place-items-center p-4 text-center text-sm text-[#ffaaa6]">
              {error}
            </div>
          )}
        </div>

        <div className="min-w-0 space-y-3">
          <label className="block text-xs font-bold text-white/65">
            Synthetic camera
            <select
              className="mt-1.5 w-full border border-white/15 bg-[#0b1712] px-2.5 py-2 text-sm text-white"
              value={cameraName}
              onChange={(event) => setCameraName(event.target.value)}
            >
              {(specification?.synthetic_camera_names ?? []).map((name) => (
                <option key={name} value={name}>{cameraLabel(name)}</option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-xs font-bold text-white/65">
            <input
              type="checkbox"
              checked={visible}
              onChange={(event) => setVisible(event.target.checked)}
            />
            Show virtual pitch
          </label>

          <label className="block text-xs font-bold text-white/65">
            Overlay opacity
            <input
              className="mt-2 w-full accent-[#9eea55]"
              type="range"
              min="0.15"
              max="1"
              step="0.05"
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
          </label>

          {preview && (
            <dl className="space-y-2 border-t border-white/10 pt-3 text-xs">
              <div>
                <dt className="text-white/35">Current camera</dt>
                <dd className="mt-0.5 font-bold">{cameraLabel(cameraName)}</dd>
              </div>
              <div>
                <dt className="text-white/35">Frame</dt>
                <dd className="mt-0.5 font-bold">
                  {preview.projection.source_camera.image_width} x{" "}
                  {preview.projection.source_camera.image_height}
                </dd>
              </div>
              <div>
                <dt className="text-white/35">Perspective check</dt>
                <dd className="mt-0.5 font-bold">
                  {preview.projection.diagnostics.perspective_order_valid
                    ? "Valid"
                    : "Review diagnostics"}
                </dd>
              </div>
            </dl>
          )}
        </div>
      </div>
    </section>
  );
}
