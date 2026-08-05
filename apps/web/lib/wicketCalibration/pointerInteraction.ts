import type { WicketBoxRole } from "./types";

export type InteractionMode =
  | "SELECT"
  | "DRAW_NEAR"
  | "DRAW_FAR"
  | "EDIT_NEAR"
  | "EDIT_FAR"
  | "LOCKED";

export type ResizeHandle = "nw" | "ne" | "sw" | "se";
export type DragMode = "move" | "resize";

export type PointerSession = {
  pointerId: number;
  role: WicketBoxRole;
  mode: DragMode;
  handle?: ResizeHandle;
};

export function canStartBoxDrag(
  mode: InteractionMode,
  role: WicketBoxRole
): boolean {
  if (mode === "LOCKED") return false;
  if (mode === "EDIT_NEAR") return role === "NEAR";
  if (mode === "EDIT_FAR") return role === "FAR";
  if (mode === "DRAW_NEAR") return role === "NEAR";
  if (mode === "DRAW_FAR") return role === "FAR";
  return false;
}

export function interactionModeAfterPointerUp(mode: InteractionMode): InteractionMode {
  if (
    mode === "EDIT_NEAR"
    || mode === "EDIT_FAR"
    || mode === "DRAW_NEAR"
    || mode === "DRAW_FAR"
  ) {
    return "SELECT";
  }
  return mode;
}

export function isDrawMode(mode: InteractionMode): boolean {
  return mode === "DRAW_NEAR" || mode === "DRAW_FAR";
}

export function shouldContinuePointerMove(buttons: number): boolean {
  return buttons !== 0;
}

export function boxCursor(
  mode: InteractionMode,
  role: WicketBoxRole,
  handle?: ResizeHandle
): string {
  if (mode === "LOCKED") return "default";
  if (!canStartBoxDrag(mode, role)) return "default";
  if (handle) return `${handle}-resize`;
  if (isDrawMode(mode)) return "crosshair";
  return "move";
}

export function resolveInteractionMode(
  boxesEditable: boolean,
  explicitMode: InteractionMode
): InteractionMode {
  if (!boxesEditable) return "LOCKED";
  if (explicitMode === "LOCKED") return "SELECT";
  return explicitMode;
}

export function matchesPointerSession(
  session: PointerSession | null,
  pointerId: number
): boolean {
  return session !== null && session.pointerId === pointerId;
}
