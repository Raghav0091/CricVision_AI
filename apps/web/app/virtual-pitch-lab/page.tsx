"use client";

import dynamic from "next/dynamic";
import {
  Component,
  type ComponentType,
  type ErrorInfo,
  type PropsWithChildren,
  useEffect,
  useMemo,
  useState
} from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { getVirtualPitchSpecification } from "@/lib/api";
import {
  adaptVirtualPitchResponse,
  calculateCameraPreset,
  materialPreset,
  resolveCameraAdjustments,
  type CameraAdjustments,
  type CameraPresetId,
  type MaterialPresetName,
  type VirtualPitchModel
} from "@/lib/virtual-pitch";
import type {
  VirtualPitchSceneProps,
  VirtualPitchVisualOptions
} from "@/components/virtual-pitch";


type PreviewLayout = "portrait" | "landscape";

type VisualOptions = {
  corridorOpacity: number;
  showPitch: boolean;
  showStumps: boolean;
  showCreases: boolean;
  showAxes: boolean;
  showGrid: boolean;
  wireframe: boolean;
  enableOrbitControls: boolean;
  lowPerformanceMode: boolean;
};

type ModelSummary = {
  version: string;
  pitchLengthM: number | null;
  pitchWidthM: number | null;
  stumpCount: number;
  lineCount: number;
  polygonCount: number;
  coordinateDescription: string;
};


const VirtualPitchCanvas = dynamic<VirtualPitchSceneProps>(
  () => import("@/components/virtual-pitch").then((module) => module.VirtualPitchCanvas as ComponentType<VirtualPitchSceneProps>),
  {
    ssr: false,
    loading: () => <ViewportMessage title="Loading 3D renderer" detail="Preparing the development scene..." />
  }
);

const PRESETS: Array<{ id: CameraPresetId; label: string }> = [
  { id: "setup", label: "Synthetic Setup Camera" },
  { id: "bowler-end", label: "Bowler-End View" },
  { id: "striker-end", label: "Striker-End View" },
  { id: "side", label: "Side View" },
  { id: "top-down", label: "Top-Down View" },
  { id: "free-orbit", label: "Free Orbit" }
];

const DEFAULT_VISUAL_OPTIONS: VisualOptions = {
  corridorOpacity: 0.24,
  showPitch: true,
  showStumps: true,
  showCreases: true,
  showAxes: false,
  showGrid: false,
  wireframe: false,
  enableOrbitControls: true,
  lowPerformanceMode: false
};


function summarizeModel(model: VirtualPitchModel | null): ModelSummary {
  return {
    version: model?.modelVersion ?? "Virtual Pitch V1",
    pitchLengthM: model?.dimensions.pitchLengthM ?? null,
    pitchWidthM: model?.dimensions.pitchWidthM ?? null,
    stumpCount: model?.stumps.length ?? 0,
    lineCount: model?.lineSegments.length ?? 0,
    polygonCount: model?.polygons.length ?? 0,
    coordinateDescription: model?.coordinateSystem.description
      ?? "Origin: bowler-end middle stump; +x pitch-right; +y toward striker; +z up."
  };
}


function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}


function formatVector(vector: { x: number; y: number; z: number }): string {
  return [vector.x, vector.y, vector.z].map((value) => value.toFixed(2)).join(", ");
}


function ViewportMessage({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="grid h-full min-h-[28rem] place-items-center bg-[#050806] p-8 text-center">
      <div className="max-w-sm">
        <p className="text-base font-bold text-white">{title}</p>
        <p className="mt-2 text-sm leading-6 text-white/45">{detail}</p>
      </div>
    </div>
  );
}


class RendererBoundary extends Component<PropsWithChildren, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(_error: Error, _info: ErrorInfo) {
    // The visible state below is the useful developer signal; Next also logs the stack.
  }

  render() {
    if (this.state.error) {
      return (
        <ViewportMessage
          title="3D renderer could not start"
          detail={this.state.error.message || "Check the renderer dependency and WebGL support."}
        />
      );
    }
    return this.props.children;
  }
}


function RangeControl({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block text-xs text-white/55">
      <span className="flex items-center justify-between gap-3">
        <span>{label}</span>
        <span className="tabular-nums text-white/80">{value.toFixed(step < 1 ? 1 : 0)}{suffix}</span>
      </span>
      <input
        className="mt-2 h-1.5 w-full cursor-pointer accent-lime"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}


function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-white/10 bg-black/15 px-3 py-2.5 text-xs font-semibold text-white/65">
      <span>{label}</span>
      <input className="h-4 w-4 accent-lime" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}


export default function VirtualPitchLabPage() {
  const [model, setModel] = useState<VirtualPitchModel | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [webglAvailable, setWebglAvailable] = useState<boolean | null>(null);
  const [layout, setLayout] = useState<PreviewLayout>("landscape");
  const [preset, setPreset] = useState<CameraPresetId>("setup");
  const [cameraAdjustments, setCameraAdjustments] = useState<CameraAdjustments>({});
  const [materialStyle, setMaterialStyle] = useState<Exclude<MaterialPresetName, "debug-wireframe">>("cricvision-dark");
  const [visualOptions, setVisualOptions] = useState<VisualOptions>(DEFAULT_VISUAL_OPTIONS);

  const aspectRatio = layout === "portrait" ? 9 / 16 : 16 / 9;
  const summary = useMemo(() => summarizeModel(model), [model]);
  const cameraPreset = useMemo(
    () => model ? calculateCameraPreset(preset, model, aspectRatio, cameraAdjustments) : null,
    [aspectRatio, cameraAdjustments, model, preset]
  );
  const displayedCamera = useMemo(
    () => model
      ? resolveCameraAdjustments(model, aspectRatio, cameraAdjustments, preset)
      : {
          heightM: 1.25,
          distanceBehindM: 3,
          lateralOffsetM: 0,
          verticalFovDegrees: 48,
          targetHeightM: 0,
          yawDegrees: 0,
          pitchDegrees: 0,
          rollDegrees: 0
        },
    [aspectRatio, cameraAdjustments, model, preset]
  );
  const rendererOptions = useMemo<VirtualPitchVisualOptions>(() => ({
    showPitch: visualOptions.showPitch,
    showStumps: visualOptions.showStumps,
    showBails: visualOptions.showStumps,
    showLines: visualOptions.showCreases,
    showCorridor: visualOptions.corridorOpacity > 0,
    showAxes: visualOptions.showAxes,
    showGrid: visualOptions.showGrid,
    enableOrbitControls: visualOptions.enableOrbitControls,
    corridorOpacity: visualOptions.corridorOpacity,
    lowPerformance: visualOptions.lowPerformanceMode,
    dprCap: visualOptions.lowPerformanceMode ? 1 : 2,
    materialPreset: materialPreset(visualOptions.wireframe ? "debug-wireframe" : materialStyle)
  }), [materialStyle, visualOptions]);
  useEffect(() => {
    setWebglAvailable(supportsWebGL());
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setLoadError(null);

    void getVirtualPitchSpecification()
      .then((response) => adaptVirtualPitchResponse(response))
      .then((nextModel) => {
        if (active) setModel(nextModel);
      })
      .catch((error: unknown) => {
        if (!active) return;
        setLoadError(error instanceof Error ? error.message : "The Virtual Pitch geometry response is invalid.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  function resetCamera() {
    setCameraAdjustments({});
  }

  function selectPreset(nextPreset: CameraPresetId) {
    setPreset(nextPreset);
    setCameraAdjustments({});
  }

  function selectLayout(nextLayout: PreviewLayout) {
    setLayout(nextLayout);
  }

  function updateCamera<K extends keyof CameraAdjustments>(key: K, value: NonNullable<CameraAdjustments[K]>) {
    setCameraAdjustments((current) => ({ ...current, [key]: value }));
  }

  function updateVisual<K extends keyof VisualOptions>(key: K, value: VisualOptions[K]) {
    setVisualOptions((current) => ({ ...current, [key]: value }));
  }

  async function reloadGeometry() {
    setLoading(true);
    setLoadError(null);
    try {
      const response = await getVirtualPitchSpecification();
      setModel(adaptVirtualPitchResponse(response));
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "The Virtual Pitch geometry response is invalid.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-[96rem] overflow-x-hidden py-1">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-white/10 pb-5">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-lime">Developer tool</p>
            <span className="rounded border border-[#ffe761]/30 bg-[#ffe761]/10 px-2 py-1 text-[10px] font-black uppercase text-[#ffe761]">Development only</span>
          </div>
          <h1 className="mt-2 text-3xl font-black sm:text-4xl">Virtual Pitch Lab</h1>
          <p className="mt-2 text-sm text-white/45">Standalone 3D inspection of the backend-owned Virtual Pitch model.</p>
        </div>
        <div className="text-left sm:text-right">
          <p className="text-xs uppercase tracking-[0.14em] text-white/35">Model</p>
          <p className="mt-1 text-sm font-bold text-white/80">{model ? summary.version : "Loading..."}</p>
        </div>
      </header>

      <div className="mt-5 grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="min-w-0 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex rounded-lg border border-white/10 bg-white/[0.03] p-1">
              {(["landscape", "portrait"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  aria-pressed={layout === option}
                  className={`rounded-md px-3 py-2 text-xs font-bold capitalize transition ${layout === option ? "bg-white/12 text-white" : "text-white/45 hover:text-white"}`}
                  onClick={() => selectLayout(option)}
                >
                  {option}
                </button>
              ))}
            </div>
            <div className="flex min-w-0 flex-1 flex-wrap justify-end gap-2">
              <select
                aria-label="Camera preset"
                className="min-w-0 rounded-lg border border-white/15 bg-[#0b1510] px-3 py-2 text-xs font-bold text-white outline-none focus:border-lime"
                value={preset}
                onChange={(event) => selectPreset(event.target.value as CameraPresetId)}
              >
                {PRESETS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
              </select>
              <Button className="px-3 py-2 text-xs" variant="secondary" onClick={() => resetCamera()}>Reset camera</Button>
            </div>
          </div>

          <div className={`mx-auto w-full overflow-hidden rounded-lg border border-white/10 bg-black shadow-glow ${layout === "portrait" ? "max-w-[32rem] aspect-[9/16]" : "aspect-video"}`}>
            {loading ? (
              <ViewportMessage title="Loading Virtual Pitch V1" detail="Requesting canonical geometry from the API..." />
            ) : loadError ? (
              <div className="grid h-full min-h-[28rem] place-items-center bg-[#100b0b] p-8 text-center">
                <div className="max-w-md">
                  <p className="text-base font-bold text-[#ffaaa6]">Virtual Pitch geometry unavailable</p>
                  <p className="mt-2 text-sm leading-6 text-white/50">{loadError}</p>
                  <Button className="mt-5" variant="secondary" onClick={() => void reloadGeometry()}>Retry API</Button>
                </div>
              </div>
            ) : webglAvailable === false ? (
              <ViewportMessage title="WebGL is unavailable" detail="Enable hardware acceleration or use a browser with WebGL support to open the 3D lab." />
            ) : model && cameraPreset ? (
              <RendererBoundary>
                <VirtualPitchCanvas model={model} mode="development" camera={cameraPreset} visualOptions={rendererOptions} />
              </RendererBoundary>
            ) : (
              <ViewportMessage title="Geometry is incomplete" detail="The API returned no validated Virtual Pitch model." />
            )}
          </div>

          <Card className="shadow-none">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/40">Scene information</p>
            <dl className="mt-4 grid gap-4 text-sm sm:grid-cols-2 lg:grid-cols-4">
              <div><dt className="text-white/35">Camera position</dt><dd className="mt-1 font-semibold tabular-nums">{cameraPreset ? formatVector(cameraPreset.position) : "-"}</dd></div>
              <div><dt className="text-white/35">Target</dt><dd className="mt-1 font-semibold tabular-nums">{cameraPreset ? formatVector(cameraPreset.target) : "-"}</dd></div>
              <div><dt className="text-white/35">Field of view</dt><dd className="mt-1 font-semibold">{cameraPreset?.verticalFovDegrees.toFixed(0) ?? "-"} degrees</dd></div>
              <div><dt className="text-white/35">Pitch dimensions</dt><dd className="mt-1 font-semibold">{summary.pitchLengthM?.toFixed(2) ?? "-"} x {summary.pitchWidthM?.toFixed(2) ?? "-"} m</dd></div>
              <div><dt className="text-white/35">Stumps</dt><dd className="mt-1 font-semibold">{summary.stumpCount}</dd></div>
              <div><dt className="text-white/35">Lines</dt><dd className="mt-1 font-semibold">{summary.lineCount}</dd></div>
              <div><dt className="text-white/35">Polygons</dt><dd className="mt-1 font-semibold">{summary.polygonCount}</dd></div>
              <div className="sm:col-span-2 lg:col-span-1"><dt className="text-white/35">Coordinates</dt><dd className="mt-1 leading-5 text-white/70">{summary.coordinateDescription}</dd></div>
            </dl>
          </Card>
        </div>

        <aside className="min-w-0 space-y-4">
          <Card className="shadow-none">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/40">Camera</p>
              <span className="text-[10px] font-bold uppercase text-[#ffe761]">Synthetic</span>
            </div>
            <div className="mt-5 space-y-5">
              <RangeControl label="Height" value={displayedCamera.heightM} min={0.2} max={8} step={0.1} suffix=" m" onChange={(value) => updateCamera("heightM", value)} />
              <RangeControl label="Distance behind wicket" value={displayedCamera.distanceBehindM} min={0.5} max={18} step={0.1} suffix=" m" onChange={(value) => updateCamera("distanceBehindM", value)} />
              <RangeControl label="Lateral offset" value={displayedCamera.lateralOffsetM} min={-8} max={8} step={0.1} suffix=" m" onChange={(value) => updateCamera("lateralOffsetM", value)} />
              <RangeControl label="Field of view" value={displayedCamera.verticalFovDegrees} min={20} max={90} step={1} suffix=" deg" onChange={(value) => updateCamera("verticalFovDegrees", value)} />
              <RangeControl label="Target height" value={displayedCamera.targetHeightM} min={0} max={4} step={0.1} suffix=" m" onChange={(value) => updateCamera("targetHeightM", value)} />
              <RangeControl label="Yaw" value={displayedCamera.yawDegrees} min={-45} max={45} step={1} suffix=" deg" onChange={(value) => updateCamera("yawDegrees", value)} />
              <RangeControl label="Pitch" value={displayedCamera.pitchDegrees} min={-45} max={45} step={1} suffix=" deg" onChange={(value) => updateCamera("pitchDegrees", value)} />
              <RangeControl label="Roll" value={displayedCamera.rollDegrees} min={-30} max={30} step={1} suffix=" deg" onChange={(value) => updateCamera("rollDegrees", value)} />
            </div>
          </Card>

          <Card className="shadow-none">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/40">Visual and debug</p>
            <div className="mt-4 space-y-2">
              <label className="block text-xs text-white/55">
                <span>Material style</span>
                <select
                  className="mt-2 w-full rounded-lg border border-white/15 bg-[#0b1510] px-3 py-2.5 text-xs font-bold text-white outline-none focus:border-lime"
                  value={materialStyle}
                  onChange={(event) => setMaterialStyle(event.target.value as Exclude<MaterialPresetName, "debug-wireframe">)}
                >
                  <option value="cricvision-dark">CricVision Dark</option>
                  <option value="broadcast-light">Broadcast Light</option>
                </select>
              </label>
              <RangeControl label="Corridor opacity" value={visualOptions.corridorOpacity} min={0} max={0.8} step={0.05} suffix="" onChange={(value) => updateVisual("corridorOpacity", value)} />
              <div className="grid grid-cols-2 gap-2 pt-2 xl:grid-cols-1 2xl:grid-cols-2">
                <Toggle label="Pitch" checked={visualOptions.showPitch} onChange={(value) => updateVisual("showPitch", value)} />
                <Toggle label="Stumps" checked={visualOptions.showStumps} onChange={(value) => updateVisual("showStumps", value)} />
                <Toggle label="Creases" checked={visualOptions.showCreases} onChange={(value) => updateVisual("showCreases", value)} />
                <Toggle label="Axes" checked={visualOptions.showAxes} onChange={(value) => updateVisual("showAxes", value)} />
                <Toggle label="Grid" checked={visualOptions.showGrid} onChange={(value) => updateVisual("showGrid", value)} />
                <Toggle label="Wireframe" checked={visualOptions.wireframe} onChange={(value) => updateVisual("wireframe", value)} />
                <Toggle label="Orbit controls" checked={visualOptions.enableOrbitControls} onChange={(value) => updateVisual("enableOrbitControls", value)} />
                <Toggle label="Low performance" checked={visualOptions.lowPerformanceMode} onChange={(value) => updateVisual("lowPerformanceMode", value)} />
              </div>
            </div>
          </Card>
        </aside>
      </div>
    </div>
  );
}
