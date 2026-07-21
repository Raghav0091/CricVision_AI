"use client";

import {
  useState,
  type CSSProperties,
  type ReactNode,
  type Ref
} from "react";


// ponytail: CSS-only fit; leave room for title/stepper/controls in the same viewport.
export const MEDIA_STAGE_MAX_H = "min(52dvh, calc(100dvh - 14rem))";
export const MEDIA_STAGE_MAX_H_EXPANDED = "min(90dvh, 56rem)";


/** Shared class for plain img/video that must fit width + viewport height. */
export const MEDIA_FIT_CLASS =
  "mx-auto block h-auto max-h-[min(42dvh,calc(100dvh-16rem))] w-auto max-w-full rounded-xl bg-black object-contain sm:max-h-[min(52dvh,calc(100dvh-14rem))]";


export const MEDIA_FIT_CLASS_EXPANDED =
  "mx-auto block h-auto max-h-[min(90dvh,56rem)] w-auto max-w-full rounded-xl bg-black object-contain";


export function mediaStageBoxStyle(
  aspectWidth: number,
  aspectHeight: number,
  expanded = false
): CSSProperties {
  const maxH = expanded ? MEDIA_STAGE_MAX_H_EXPANDED : MEDIA_STAGE_MAX_H;
  const safeW = Math.max(aspectWidth, 1);
  const safeH = Math.max(aspectHeight, 1);
  return {
    aspectRatio: `${safeW} / ${safeH}`,
    width: `min(100%, calc((${maxH}) * ${safeW} / ${safeH}))`,
    maxHeight: maxH,
    height: "auto"
  };
}


/**
 * Viewport-bounded media stage. Letterboxes portrait/landscape inside a neutral frame.
 * Children fill the aspect box; use object-fit: contain on media (never cover).
 */
export function AnalysisMediaStage({
  children,
  aspectWidth,
  aspectHeight,
  label,
  className = "",
  expandable = false,
  stageRef
}: {
  children: ReactNode;
  aspectWidth?: number;
  aspectHeight?: number;
  label?: string;
  className?: string;
  expandable?: boolean;
  stageRef?: Ref<HTMLDivElement>;
}) {
  const [expanded, setExpanded] = useState(false);
  const maxH = expanded ? MEDIA_STAGE_MAX_H_EXPANDED : MEDIA_STAGE_MAX_H;
  const hasAspect =
    aspectWidth != null
    && aspectHeight != null
    && aspectWidth > 0
    && aspectHeight > 0;

  return (
    <div className={`relative ${className}`}>
      {expandable && (
        <div className="mb-2 flex items-center justify-end gap-2">
          <button
            type="button"
            className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] text-white/65 hover:bg-white/10"
            aria-pressed={expanded}
            aria-label={expanded ? "Fit media to workspace" : "Expand media"}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "Fit" : "Expand"}
          </button>
        </div>
      )}
      <div
        className="flex w-full items-center justify-center overflow-hidden rounded-xl bg-[#050a08]"
        style={{ maxHeight: maxH, minHeight: "10rem" }}
        role="group"
        aria-label={label}
      >
        {hasAspect ? (
          <div
            ref={stageRef}
            className="relative max-w-full overflow-hidden"
            style={mediaStageBoxStyle(aspectWidth, aspectHeight, expanded)}
          >
            {children}
          </div>
        ) : (
          <div
            ref={stageRef}
            className="relative w-full max-w-full"
            style={{ maxHeight: maxH }}
          >
            {children}
          </div>
        )}
      </div>
    </div>
  );
}
