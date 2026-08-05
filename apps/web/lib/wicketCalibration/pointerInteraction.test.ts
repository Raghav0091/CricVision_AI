import {
  boxCursor,
  canStartBoxDrag,
  interactionModeAfterPointerUp,
  matchesPointerSession,
  resolveInteractionMode,
  shouldContinuePointerMove,
  type PointerSession
} from "./pointerInteraction";


function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}


assert(canStartBoxDrag("EDIT_NEAR", "NEAR"), "EDIT_NEAR allows NEAR drag");
assert(!canStartBoxDrag("EDIT_NEAR", "FAR"), "EDIT_NEAR blocks FAR drag");
assert(!canStartBoxDrag("SELECT", "NEAR"), "SELECT blocks drag");
assert(!canStartBoxDrag("LOCKED", "NEAR"), "LOCKED blocks drag");
assert(canStartBoxDrag("DRAW_FAR", "FAR"), "DRAW_FAR allows FAR drag");

assert(interactionModeAfterPointerUp("EDIT_NEAR") === "SELECT", "pointer up returns SELECT");
assert(interactionModeAfterPointerUp("DRAW_FAR") === "SELECT", "draw completes to SELECT");
assert(interactionModeAfterPointerUp("LOCKED") === "LOCKED", "LOCKED stays LOCKED");

assert(resolveInteractionMode(false, "EDIT_NEAR") === "LOCKED", "non-editable locks");
assert(resolveInteractionMode(true, "EDIT_NEAR") === "EDIT_NEAR", "editable keeps mode");

assert(shouldContinuePointerMove(1), "button pressed continues drag");
assert(!shouldContinuePointerMove(0), "no buttons stops drag");

const session: PointerSession = { pointerId: 7, role: "NEAR", mode: "move" };
assert(matchesPointerSession(session, 7), "matching pointer id");
assert(!matchesPointerSession(session, 8), "non-matching pointer id");
assert(!matchesPointerSession(null, 7), "null session");

assert(boxCursor("LOCKED", "NEAR") === "default", "locked default cursor");
assert(boxCursor("EDIT_NEAR", "NEAR", "se") === "se-resize", "resize handle cursor");
assert(boxCursor("DRAW_NEAR", "NEAR") === "crosshair", "draw crosshair");
assert(boxCursor("SELECT", "NEAR") === "default", "select default cursor");

console.log("pointerInteraction.test.ts passed");
