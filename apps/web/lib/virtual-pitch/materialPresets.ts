import type { MaterialStyle, VirtualPitchMaterialPreset } from "./types";

function material(
  color: string,
  options: Partial<Omit<MaterialStyle, "color">> = {}
): MaterialStyle {
  const opacity = options.opacity ?? 1;
  return Object.freeze({
    color,
    opacity,
    roughness: options.roughness ?? 0.75,
    metalness: options.metalness ?? 0,
    wireframe: options.wireframe ?? false,
    transparent: options.transparent ?? opacity < 1,
    depthWrite: options.depthWrite ?? opacity >= 1
  });
}

export const MATERIAL_PRESETS = Object.freeze({
  "cricvision-dark": Object.freeze({
    id: "cricvision-dark",
    label: "CricVision Dark",
    background: "#101418",
    ambientLight: "#cbd5e1",
    keyLight: "#fff7d6",
    pitch: material("#64705b", { roughness: 0.95 }),
    stump: material("#e8b84a", { roughness: 0.55 }),
    bail: material("#f4cf72", { roughness: 0.5 }),
    officialLine: material("#f8fafc", { roughness: 0.8 }),
    analyticalLine: material("#9cc7ff", { opacity: 0.9 }),
    corridor: material("#7766d7", { opacity: 0.2, depthWrite: false })
  }),
  "broadcast-light": Object.freeze({
    id: "broadcast-light",
    label: "Broadcast Light",
    background: "#dce7ee",
    ambientLight: "#ffffff",
    keyLight: "#fff2cc",
    pitch: material("#7e9a6d", { roughness: 0.9 }),
    stump: material("#d6a72c", { roughness: 0.55 }),
    bail: material("#e4bd4d", { roughness: 0.5 }),
    officialLine: material("#ffffff"),
    analyticalLine: material("#315a8a", { opacity: 0.85 }),
    corridor: material("#6355ba", { opacity: 0.16, depthWrite: false })
  }),
  "debug-wireframe": Object.freeze({
    id: "debug-wireframe",
    label: "Debug Wireframe",
    background: "#080b0f",
    ambientLight: "#ffffff",
    keyLight: "#ffffff",
    pitch: material("#48d7a1", { opacity: 0.35, wireframe: true, depthWrite: false }),
    stump: material("#ffcc44", { wireframe: true }),
    bail: material("#ffea8a", { wireframe: true }),
    officialLine: material("#ffffff", { wireframe: true }),
    analyticalLine: material("#4fb3ff", { wireframe: true }),
    corridor: material("#c084fc", { opacity: 0.28, wireframe: true, depthWrite: false })
  })
} satisfies Record<string, VirtualPitchMaterialPreset>);

export type MaterialPresetName = keyof typeof MATERIAL_PRESETS;

export function materialPreset(id: MaterialPresetName): VirtualPitchMaterialPreset {
  return MATERIAL_PRESETS[id];
}
