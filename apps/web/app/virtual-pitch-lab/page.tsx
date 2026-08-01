"use client";

import dynamic from "next/dynamic";
import {
  Component,
  type ComponentType,
  type ErrorInfo,
  type PropsWithChildren,
  useCallback,
  useEffect,
  useMemo,
  useState
} from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { AssistedSceneCalibrationPanel } from "@/components/video-analysis/AssistedSceneCalibrationPanel";
import type {
  ActiveCameraDiagnostics,
  VirtualPitchSceneProps,
  VirtualPitchVisualOptions
} from "@/components/virtual-pitch";
import {
  AutoRegistrationPanel,
  CameraDiagnosticsPanel,
  OverlayStage,
  type CameraBridgePayload,
  type CameraSourceMode,
  type OverlayComparisonMode
} from "@/components/virtual-pitch-lab";
import {
  autoRegistrationCameraBridge,
  clearPresetAutoRegistration,
  getAnalysisCameraBridge,
  getCameraSetupPresets,
  getPresetAutoRegistration,
  getSyntheticCameraBridge,
  getVirtualPitchSpecification,
  runPresetAutoRegistration,
  type CameraSetupPreset,
  type NormalizedCameraBridgeResponse,
  type PitchProjectionGeometry,
  type PresetAutoRegistrationResult,
} from "@/lib/api";
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
import { validateCameraBridge } from "@/lib/virtual-pitch/cameraValidation";
import { buildThreeCameraFromOpenCv, type ThreeCameraBridge } from "@/lib/virtual-pitch/opencvCameraBridge";


const STRONGEST_ANALYSIS_ID = "analysis_20260728_120858_762989";

type PreviewLayout = "portrait" | "landscape";
type LabCanvasProps = VirtualPitchSceneProps & {
  calibratedCamera?: ThreeCameraBridge;
};

type VisualOptions = {
  corridorOpacity: number;
  showPitch: boolean;
  showStumps: boolean;
  showCreases: boolean;
  showCorridor: boolean;
  showAxes: boolean;
  showGrid: boolean;
  wireframe: boolean;
  enableOrbitControls: boolean;
  lowPerformanceMode: boolean;
};


const VirtualPitchCanvas = dynamic<LabCanvasProps>(
  () => import("@/components/virtual-pitch").then((module) => module.VirtualPitchCanvas as ComponentType<LabCanvasProps>),
  { ssr: false, loading: () => <ViewportMessage title="Loading 3D renderer" detail="Preparing the development scene..." /> }
);

const PRESETS: Array<{ id: CameraPresetId; label: string }> = [
  { id: "setup", label: "Synthetic Setup Camera" },
  { id: "bowler-end", label: "Bowler-End View" },
  { id: "striker-end", label: "Striker-End View" },
  { id: "side", label: "Side View" },
  { id: "top-down", label: "Top-Down View" },
  { id: "free-orbit", label: "Free Orbit" }
];

const SOURCE_OPTIONS: Array<{ id: CameraSourceMode; label: string }> = [
  { id: "auto-registration", label: "One-click Auto Registration" },
  { id: "development", label: "Development Preset" },
  { id: "synthetic-opencv", label: "Synthetic OpenCV Camera" },
  { id: "real-analysis", label: "Real Analysis Camera" }
];

const DEFAULT_VISUAL_OPTIONS: VisualOptions = {
  corridorOpacity: 0.24,
  showPitch: true,
  showStumps: true,
  showCreases: true,
  showCorridor: true,
  showAxes: false,
  showGrid: false,
  wireframe: false,
  enableOrbitControls: true,
  lowPerformanceMode: false
};


function comparisonsFromProjection(projection?: PitchProjectionGeometry | null) {
  return (projection?.projected_landmarks ?? []).map((landmark) => ({
    semantic_id: landmark.semantic_id,
    world_point: landmark.world_point,
    opencv_pixel: landmark.pixel_point,
    three_pixel: null,
    camera_depth: landmark.depth_m,
    in_frame: landmark.in_frame,
    clipped: !landmark.projection_valid
  }));
}


function cameraLabel(name: string): string {
  return name.split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}


function supportsWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}


function ViewportMessage({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="grid h-full min-h-[22rem] place-items-center bg-[#050806] p-8 text-center">
      <div className="max-w-sm"><p className="text-base font-bold text-white">{title}</p><p className="mt-2 text-sm leading-6 text-white/45">{detail}</p></div>
    </div>
  );
}


class RendererBoundary extends Component<PropsWithChildren, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(_error: Error, _info: ErrorInfo) {}
  render() {
    return this.state.error
      ? <ViewportMessage title="3D renderer could not start" detail={this.state.error.message || "Check WebGL support."} />
      : this.props.children;
  }
}


function RangeControl({ label, value, min, max, step, suffix, disabled = false, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className={`block text-xs ${disabled ? "text-white/25" : "text-white/55"}`}>
      <span className="flex items-center justify-between gap-3"><span>{label}</span><span className="tabular-nums">{value.toFixed(step < 1 ? 1 : 0)}{suffix}</span></span>
      <input className="mt-2 h-1.5 w-full accent-lime disabled:cursor-not-allowed disabled:opacity-30" disabled={disabled} type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}


function Toggle({ label, checked, disabled = false, onChange }: { label: string; checked: boolean; disabled?: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className={`flex items-center justify-between gap-3 rounded border border-white/10 bg-black/15 px-3 py-2.5 text-xs font-semibold ${disabled ? "cursor-not-allowed text-white/25" : "cursor-pointer text-white/65"}`}>
      <span>{label}</span><input className="h-4 w-4 accent-lime" disabled={disabled} type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}


export default function VirtualPitchLabPage() {
  const [model, setModel] = useState<VirtualPitchModel | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [webglAvailable, setWebglAvailable] = useState<boolean | null>(null);
  const [sourceMode, setSourceMode] = useState<CameraSourceMode>("auto-registration");
  const [layout, setLayout] = useState<PreviewLayout>("landscape");
  const [preset, setPreset] = useState<CameraPresetId>("setup");
  const [cameraAdjustments, setCameraAdjustments] = useState<CameraAdjustments>({});
  const [materialStyle, setMaterialStyle] = useState<Exclude<MaterialPresetName, "debug-wireframe">>("cricvision-dark");
  const [visualOptions, setVisualOptions] = useState<VisualOptions>(DEFAULT_VISUAL_OPTIONS);
  const [syntheticCameraName, setSyntheticCameraName] = useState("");
  const [syntheticPayload, setSyntheticPayload] = useState<NormalizedCameraBridgeResponse | null>(null);
  const [analysisId, setAnalysisId] = useState(STRONGEST_ANALYSIS_ID);
  const [setupPresets, setSetupPresets] = useState<CameraSetupPreset[]>([]);
  const [setupPresetId, setSetupPresetId] = useState("");
  const [autoResult, setAutoResult] = useState<PresetAutoRegistrationResult | null>(null);
  const [autoBusy, setAutoBusy] = useState(false);
  const [autoError, setAutoError] = useState<string | null>(null);
  const [presetsLoading, setPresetsLoading] = useState(true);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [bridgePayload, setBridgePayload] = useState<CameraBridgePayload | null>(null);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [comparisonMode, setComparisonMode] = useState<OverlayComparisonMode>("both");
  const [overlayOpacity, setOverlayOpacity] = useState(0.76);
  const [showOpenCvMarkers, setShowOpenCvMarkers] = useState(true);
  const [showThreeMarkers, setShowThreeMarkers] = useState(true);
  const [showResiduals, setShowResiduals] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [showCameraDiagnostics, setShowCameraDiagnostics] = useState(true);
  const [activeCameraDiagnostics, setActiveCameraDiagnostics] = useState<ActiveCameraDiagnostics | null>(null);
  const [activeCanvasCount, setActiveCanvasCount] = useState(0);

  const calibrated = sourceMode !== "development";
  const aspectRatio = layout === "portrait" ? 9 / 16 : 16 / 9;
  const cameraPreset = useMemo(() => model ? calculateCameraPreset(preset, model, aspectRatio, cameraAdjustments) : null, [aspectRatio, cameraAdjustments, model, preset]);
  const displayedCamera = useMemo(() => model ? resolveCameraAdjustments(model, aspectRatio, cameraAdjustments, preset) : null, [aspectRatio, cameraAdjustments, model, preset]);
  const automaticPayload = useMemo(() => autoRegistrationCameraBridge(autoResult), [autoResult]);
  const activePayload = sourceMode === "auto-registration" && automaticPayload
    ? { ...automaticPayload, comparisons: comparisonsFromProjection(automaticPayload.projection), diagnostics: null } satisfies CameraBridgePayload
    : sourceMode === "synthetic-opencv" && syntheticPayload
    ? { ...syntheticPayload, comparisons: comparisonsFromProjection(syntheticPayload.projection), diagnostics: null } satisfies CameraBridgePayload
    : sourceMode === "real-analysis" ? bridgePayload : null;
  const threeBridge = useMemo(() => {
    if (!activePayload) return null;
    try {
      return buildThreeCameraFromOpenCv(activePayload.camera);
    } catch {
      return null;
    }
  }, [activePayload]);
  const validationReport = useMemo(() => {
    if (!threeBridge || !model) return null;
    return validateCameraBridge(
      threeBridge,
      model.landmarks.map((landmark) => ({ semanticId: landmark.semanticId, world: landmark.point }))
    );
  }, [model, threeBridge]);
  const nativeWidth = activePayload?.camera.image_width ?? 1280;
  const nativeHeight = activePayload?.camera.image_height ?? 720;
  const comparisons = validationReport?.points.map((point) => ({
    semantic_id: point.semanticId,
    world_point: point.world,
    opencv_pixel: point.openCvPixel,
    three_pixel: point.threePixel,
    residual_x_px: point.xResidual,
    residual_y_px: point.yResidual,
    error_px: point.pixelError,
    camera_depth: point.cameraDepth,
    in_frame: point.inFrame,
    clipped: point.clippingState !== "inside_frustum"
  })) ?? activePayload?.comparisons ?? comparisonsFromProjection(activePayload?.projection);
  const diagnostics = validationReport && threeBridge ? {
    point_count: validationReport.metrics.pointCount,
    valid_point_count: validationReport.metrics.validPointCount,
    invalid_point_count: validationReport.metrics.invalidPointCount,
    points_behind_camera: validationReport.metrics.pointsBehindCamera,
    rmse_px: validationReport.metrics.rmse,
    maximum_error_px: validationReport.metrics.maximumError,
    mean_error_px: validationReport.metrics.meanError,
    median_error_px: validationReport.metrics.medianError,
    horizontal_bias_px: validationReport.metrics.horizontalBias,
    vertical_bias_px: validationReport.metrics.verticalBias,
    finite_matrices: validationReport.metrics.finiteMatrixStatus,
    mirrored_axis_warning: validationReport.metrics.mirroredAxisWarning,
    bowler_striker_reversal_warning: validationReport.metrics.bowlerStrikerReversalWarning,
    distortion_mode: threeBridge.diagnostics.distortion.mode,
    exact: threeBridge.exact,
    warnings: threeBridge.diagnostics.warnings
  } as const : activePayload?.diagnostics;
  const rendererOptions = useMemo<VirtualPitchVisualOptions>(() => ({
    showPitch: visualOptions.showPitch,
    showStumps: visualOptions.showStumps,
    showBails: visualOptions.showStumps,
    showLines: visualOptions.showCreases,
    showCorridor: visualOptions.showCorridor && visualOptions.corridorOpacity > 0,
    showAxes: !calibrated && visualOptions.showAxes,
    showGrid: !calibrated && visualOptions.showGrid,
    enableOrbitControls: !calibrated && visualOptions.enableOrbitControls,
    corridorOpacity: visualOptions.corridorOpacity,
    lowPerformance: visualOptions.lowPerformanceMode,
    dprCap: visualOptions.lowPerformanceMode ? 1 : 2,
    materialPreset: materialPreset(visualOptions.wireframe ? "debug-wireframe" : materialStyle)
  }), [calibrated, materialStyle, visualOptions]);

  useEffect(() => setWebglAvailable(supportsWebGL()), []);
  useEffect(() => {
    let active = true;
    setPresetsLoading(true);
    getCameraSetupPresets()
      .then((presets) => {
        if (!active) return;
        setSetupPresets(presets);
        setSetupPresetId((current) => current || presets[0]?.preset_id || "");
      })
      .catch((error: unknown) => active && setAutoError(error instanceof Error ? error.message : "Camera setup presets are unavailable."))
      .finally(() => active && setPresetsLoading(false));
    return () => { active = false; };
  }, []);
  useEffect(() => {
    let active = true;
    setLoading(true);
    getVirtualPitchSpecification()
      .then((response) => {
        if (!active) return;
        setModel(adaptVirtualPitchResponse(response));
        setSyntheticCameraName(response.synthetic_camera_names[0] ?? "");
      })
      .catch((error: unknown) => active && setLoadError(error instanceof Error ? error.message : "Virtual Pitch geometry is unavailable."))
      .finally(() => active && setLoading(false));
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (sourceMode !== "synthetic-opencv" || !syntheticCameraName) return;
    let active = true;
    setCameraLoading(true);
    setCameraError(null);
    void getSyntheticCameraBridge(syntheticCameraName)
      .then((payload) => active && setSyntheticPayload(payload))
      .catch((error: unknown) => active && setCameraError(error instanceof Error ? error.message : "Synthetic camera is unavailable."))
      .finally(() => active && setCameraLoading(false));
    return () => { active = false; };
  }, [sourceMode, syntheticCameraName]);

  async function loadRealCamera() {
    setCameraLoading(true);
    setCameraError(null);
    setBridgePayload(null);
    try {
      setBridgePayload(await getAnalysisCameraBridge(analysisId.trim()));
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : "Real camera candidate is unavailable.");
    } finally {
      setCameraLoading(false);
    }
  }

  async function runAutomaticRegistration() {
    const id = analysisId.trim();
    if (!id || !setupPresetId) return;
    setAutoBusy(true);
    setAutoError(null);
    try {
      const result = await runPresetAutoRegistration(id, {
        preset_id: setupPresetId,
        reuse_existing_observations: true,
        force_redetect: false,
        development_diagnostics: true
      });
      setAutoResult(result);
      if (result.status === "AUTO_REGISTRATION_READY" || result.status === "VISUAL_OVERLAY_READY") {
        setAdvancedOpen(false);
      } else if (result.manual_assistance_available) {
        setAdvancedOpen(true);
      }
    } catch (error) {
      setAutoError(error instanceof Error ? error.message : "Automatic registration failed.");
    } finally {
      setAutoBusy(false);
    }
  }

  async function clearAutomaticRegistration() {
    const id = analysisId.trim();
    if (!id) return;
    setAutoBusy(true);
    setAutoError(null);
    try {
      await clearPresetAutoRegistration(id);
      setAutoResult(null);
    } catch (error) {
      setAutoError(error instanceof Error ? error.message : "Automatic registration result could not be cleared.");
    } finally {
      setAutoBusy(false);
    }
  }

  useEffect(() => {
    if (sourceMode === "real-analysis") void loadRealCamera();
    // Loading is intentionally tied to source selection; subsequent IDs use the explicit button.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceMode]);

  useEffect(() => {
    if (sourceMode !== "auto-registration" || !analysisId.trim()) return;
    let active = true;
    setAutoError(null);
    void getPresetAutoRegistration(analysisId.trim())
      .then((result) => active && setAutoResult(result))
      .catch((error: unknown) => active && setAutoError(error instanceof Error ? error.message : "Saved automatic registration could not be loaded."));
    return () => { active = false; };
    // Existing results are restored on mode entry; editing the ID does not issue requests per keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceMode]);

  function updateVisual<K extends keyof VisualOptions>(key: K, value: VisualOptions[K]) {
    setVisualOptions((current) => ({ ...current, [key]: value }));
  }

  const handleCameraDiagnostics = useCallback((value: ActiveCameraDiagnostics) => {
    setActiveCameraDiagnostics(value);
  }, []);
  const handleCanvasCountChange = useCallback((count: number) => setActiveCanvasCount(count), []);

  const canvas = model && cameraPreset ? (
    <RendererBoundary>
      <VirtualPitchCanvas
        model={model}
        mode={(calibrated ? (sourceMode === "real-analysis" || sourceMode === "auto-registration" ? "real-frame-overlay" : "camera-validation") : "development") as VirtualPitchSceneProps["mode"]}
        camera={cameraPreset}
        calibratedCamera={threeBridge ?? undefined}
        visualOptions={rendererOptions}
        onCameraDiagnostics={handleCameraDiagnostics}
      />
    </RendererBoundary>
  ) : null;

  return (
    <div className="mx-auto max-w-[96rem] overflow-x-hidden py-1">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-white/10 pb-5">
        <div>
          <div className="flex flex-wrap items-center gap-2"><p className="text-xs font-bold uppercase tracking-[0.18em] text-lime">Developer tool</p><span className="rounded border border-[#ffe761]/30 bg-[#ffe761]/10 px-2 py-1 text-[10px] font-black uppercase text-[#ffe761]">Development only</span></div>
          <h1 className="mt-2 text-3xl font-black sm:text-4xl">Virtual Pitch Lab</h1>
          <p className="mt-2 text-sm text-white/45">OpenCV-to-Three.js camera validation and real-frame overlay inspection.</p>
        </div>
        <p className="text-sm font-bold text-white/70">{model?.modelVersion ?? "Loading Virtual Pitch V1..."}</p>
      </header>

      <div className="mt-5 flex gap-2 border-b border-white/10 pb-4">
        <button type="button" aria-pressed={sourceMode === "auto-registration"} className={`px-3 py-2 text-xs font-bold ${sourceMode === "auto-registration" ? "bg-white/12 text-white" : "text-white/45 hover:text-white"}`} onClick={() => { setSourceMode("auto-registration"); setAdvancedOpen(false); }}>One-click Auto Registration</button>
        <button type="button" aria-expanded={advancedOpen} className={`px-3 py-2 text-xs font-bold ${sourceMode !== "auto-registration" || advancedOpen ? "text-white" : "text-white/45 hover:text-white"}`} onClick={() => { if (advancedOpen && sourceMode !== "auto-registration") setSourceMode("auto-registration"); setAdvancedOpen((open) => !open); }}>Advanced Calibration {advancedOpen ? "-" : "+"}</button>
      </div>

      {advancedOpen && <div className="mt-3 flex overflow-x-auto rounded border border-white/10 bg-white/[0.03] p-1">{SOURCE_OPTIONS.filter((source) => source.id !== "auto-registration").map((source) => <button key={source.id} type="button" aria-pressed={sourceMode === source.id} className={`min-w-fit flex-1 px-3 py-2 text-xs font-bold transition ${sourceMode === source.id ? "bg-white/12 text-white" : "text-white/45 hover:text-white"}`} onClick={() => setSourceMode(source.id)}>{source.label}</button>)}</div>}

      <div className="mt-5 grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_21rem]">
        <main className="min-w-0 space-y-4">
          {sourceMode === "auto-registration" && <AutoRegistrationPanel
            analysisId={analysisId}
            presetId={setupPresetId}
            presets={setupPresets}
            result={autoResult}
            busy={autoBusy}
            loadingPresets={presetsLoading}
            error={autoError}
            onAnalysisIdChange={(value) => { setAnalysisId(value); setAutoResult(null); }}
            onPresetIdChange={(value) => { setSetupPresetId(value); setAutoResult(null); }}
            onRun={() => void runAutomaticRegistration()}
            onClear={() => void clearAutomaticRegistration()}
            onOpenAdvanced={() => setAdvancedOpen(true)}
          />}

          {sourceMode !== "auto-registration" && <div className="flex flex-wrap items-center justify-between gap-3">
            {sourceMode === "development" ? (
              <>
                <div className="flex rounded border border-white/10 bg-white/[0.03] p-1">
                  {(["landscape", "portrait"] as const).map((option) => <button key={option} type="button" className={`px-3 py-2 text-xs font-bold capitalize ${layout === option ? "bg-white/12 text-white" : "text-white/45"}`} onClick={() => setLayout(option)}>{option}</button>)}
                </div>
                <select aria-label="Camera preset" className="rounded border border-white/15 bg-[#0b1510] px-3 py-2 text-xs font-bold text-white" value={preset} onChange={(event) => { setPreset(event.target.value as CameraPresetId); setCameraAdjustments({}); }}>{PRESETS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</select>
              </>
            ) : sourceMode === "synthetic-opencv" ? (
              <select aria-label="Synthetic OpenCV camera" className="rounded border border-white/15 bg-[#0b1510] px-3 py-2 text-xs font-bold text-white" value={syntheticCameraName} onChange={(event) => setSyntheticCameraName(event.target.value)}>{(model?.syntheticCameraNames ?? []).map((name) => <option key={name} value={name}>{cameraLabel(name)}</option>)}</select>
            ) : (
              <div className="flex min-w-0 flex-1 gap-2"><input aria-label="Analysis ID" className="min-w-0 flex-1 rounded border border-white/15 bg-[#0b1510] px-3 py-2 text-xs text-white" value={analysisId} onChange={(event) => setAnalysisId(event.target.value)} /><Button className="px-3 py-2 text-xs" disabled={cameraLoading || !analysisId.trim()} variant="secondary" onClick={() => void loadRealCamera()}>Load</Button></div>
            )}
            {calibrated && (
              <select aria-label="Overlay comparison" className="rounded border border-white/15 bg-[#0b1510] px-3 py-2 text-xs font-bold text-white" value={comparisonMode} onChange={(event) => setComparisonMode(event.target.value as OverlayComparisonMode)}>
                <option value="three">Three.js only</option><option value="svg">OpenCV SVG only</option><option value="both">Both overlays</option><option value="side-by-side">Side by side</option>
              </select>
            )}
          </div>}

          <div className={`overflow-hidden rounded border border-white/10 bg-black shadow-glow ${sourceMode === "development" ? (layout === "portrait" ? "mx-auto aspect-[9/16] w-full max-w-[32rem]" : "aspect-video") : "h-[min(68dvh,48rem)] min-h-[22rem]"}`}>
            {loading || cameraLoading ? <ViewportMessage title="Loading camera stage" detail="Preparing canonical geometry and camera evidence..." />
              : loadError || cameraError ? <ViewportMessage title="Camera stage unavailable" detail={loadError ?? cameraError ?? "Unknown camera error."} />
              : webglAvailable === false ? <ViewportMessage title="WebGL is unavailable" detail="Enable hardware acceleration to inspect the 3D overlay." />
              : sourceMode === "development" ? (canvas ?? <ViewportMessage title="Geometry is incomplete" detail="No validated Virtual Pitch model was returned." />)
              : activePayload && canvas ? (
                <OverlayStage
                  imageWidth={nativeWidth}
                  imageHeight={nativeHeight}
                  frameUrl={sourceMode === "real-analysis" || sourceMode === "auto-registration" ? activePayload.camera.setup_frame_url : null}
                  threeCanvas={canvas}
                  projection={activePayload.projection}
                  comparisons={comparisons}
                  comparisonMode={comparisonMode}
                  overlayOpacity={overlayOpacity}
                  showOpenCvMarkers={showOpenCvMarkers}
                  showThreeMarkers={showThreeMarkers}
                  showResiduals={showResiduals}
                  showLabels={showLabels}
                  onCanvasCountChange={handleCanvasCountChange}
                />
              ) : <ViewportMessage title={sourceMode === "auto-registration" ? "No automatic alignment yet" : "No calibrated camera"} detail={sourceMode === "auto-registration" ? "Choose a setup preset, then run Auto Detect and Align." : "Select a synthetic camera or load a real analysis camera candidate."} />}
          </div>

          {calibrated && (
            <div className="flex flex-wrap gap-x-5 gap-y-2 border border-white/10 bg-white/[0.02] px-3 py-2 text-[11px] text-white/60">
              <span><span className="mr-1 inline-block h-2.5 w-2.5 rounded-full border-2 border-[#ffe56b]" /> OpenCV marker</span>
              <span><span className="mr-1 font-black text-[#5ee7ff]">+</span> Three.js marker</span>
              <span><span className="mr-1 inline-block h-px w-4 bg-[#ff5f87] align-middle" /> Residual</span>
              <span className="ml-auto tabular-nums">Native {nativeWidth} x {nativeHeight}</span>
            </div>
          )}
        </main>

        <aside className="min-w-0 space-y-4">
          {sourceMode !== "auto-registration" && <Card className="shadow-none">
            <div className="flex items-center justify-between gap-3"><p className="text-xs font-bold uppercase tracking-[0.14em] text-white/40">Camera</p><span className="text-[10px] font-bold uppercase text-[#ffe761]">{calibrated ? "Calibrated" : "Development"}</span></div>
            {displayedCamera && <div className="mt-5 space-y-5">
              <RangeControl disabled={calibrated} label="Height" value={displayedCamera.heightM} min={0.2} max={8} step={0.1} suffix=" m" onChange={(value) => setCameraAdjustments((current) => ({ ...current, heightM: value }))} />
              <RangeControl disabled={calibrated} label="Distance behind wicket" value={displayedCamera.distanceBehindM} min={0.5} max={18} step={0.1} suffix=" m" onChange={(value) => setCameraAdjustments((current) => ({ ...current, distanceBehindM: value }))} />
              <RangeControl disabled={calibrated} label="Field of view" value={displayedCamera.verticalFovDegrees} min={20} max={90} step={1} suffix=" deg" onChange={(value) => setCameraAdjustments((current) => ({ ...current, verticalFovDegrees: value }))} />
              <Toggle disabled={calibrated} label="Orbit controls" checked={!calibrated && visualOptions.enableOrbitControls} onChange={(value) => updateVisual("enableOrbitControls", value)} />
              {!calibrated && <Button className="w-full text-xs" variant="secondary" onClick={() => setCameraAdjustments({})}>Reset camera</Button>}
            </div>}
          </Card>}

          <Card className="shadow-none">
            <p className="text-xs font-bold uppercase tracking-[0.14em] text-white/40">Visual and debug</p>
            <div className="mt-4 space-y-3">
              <RangeControl label="Overlay opacity" value={overlayOpacity} min={0.1} max={1} step={0.05} suffix="" onChange={setOverlayOpacity} />
              <label className="block text-xs text-white/55">Material style<select className="mt-2 w-full rounded border border-white/15 bg-[#0b1510] px-3 py-2.5 text-xs font-bold text-white" value={materialStyle} onChange={(event) => setMaterialStyle(event.target.value as Exclude<MaterialPresetName, "debug-wireframe">)}><option value="cricvision-dark">CricVision Dark</option><option value="broadcast-light">Broadcast Light</option></select></label>
              <RangeControl label="Corridor opacity" value={visualOptions.corridorOpacity} min={0} max={0.8} step={0.05} suffix="" onChange={(value) => updateVisual("corridorOpacity", value)} />
              <div className="grid grid-cols-2 gap-2 xl:grid-cols-1 2xl:grid-cols-2">
                <Toggle label="Pitch" checked={visualOptions.showPitch} onChange={(value) => updateVisual("showPitch", value)} />
                <Toggle label="Stumps" checked={visualOptions.showStumps} onChange={(value) => updateVisual("showStumps", value)} />
                <Toggle label="Creases" checked={visualOptions.showCreases} onChange={(value) => updateVisual("showCreases", value)} />
                <Toggle label="Corridor" checked={visualOptions.showCorridor} onChange={(value) => updateVisual("showCorridor", value)} />
                {calibrated && <><Toggle label="OpenCV markers" checked={showOpenCvMarkers} onChange={setShowOpenCvMarkers} /><Toggle label="Three markers" checked={showThreeMarkers} onChange={setShowThreeMarkers} /><Toggle label="Residual lines" checked={showResiduals} onChange={setShowResiduals} /><Toggle label="Marker labels" checked={showLabels} onChange={setShowLabels} /><Toggle label="Camera diagnostics" checked={showCameraDiagnostics} onChange={setShowCameraDiagnostics} /></>}
                {!calibrated && <><Toggle label="Axes" checked={visualOptions.showAxes} onChange={(value) => updateVisual("showAxes", value)} /><Toggle label="Grid" checked={visualOptions.showGrid} onChange={(value) => updateVisual("showGrid", value)} /></>}
                <Toggle label="Wireframe" checked={visualOptions.wireframe} onChange={(value) => updateVisual("wireframe", value)} />
                <Toggle label="Low performance" checked={visualOptions.lowPerformanceMode} onChange={(value) => updateVisual("lowPerformanceMode", value)} />
              </div>
            </div>
          </Card>

          {calibrated && showCameraDiagnostics && <Card className="shadow-none"><p className="mb-4 text-xs font-bold uppercase tracking-[0.14em] text-white/40">Bridge diagnostics</p><CameraDiagnosticsPanel camera={activePayload?.camera ?? null} diagnostics={diagnostics} activeCamera={activeCameraDiagnostics} requestedCamera={sourceMode} activeCanvasCount={activeCanvasCount} /></Card>}
        </aside>
      </div>

      {sourceMode === "auto-registration" && advancedOpen && analysisId.trim() && <section className="mt-5 border border-white/10 bg-white/[0.02] p-4 sm:p-5"><div className="mb-4"><p className="text-xs font-black uppercase text-[#ffe761]">Advanced Calibration</p><p className="mt-1 text-sm text-white/45">Manual anchors remain separate from the automatic result and are changed only by your actions here.</p></div><AssistedSceneCalibrationPanel key={analysisId.trim()} analysisId={analysisId.trim()} /></section>}
    </div>
  );
}
