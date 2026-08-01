# Wicket Landmark Evidence V1 - Virtual Pitch Lab QA

Date: 2026-08-02
Scope: developer-only Virtual Pitch Lab diagnostics owned by Agent 4
Baseline HEAD: `517c42a45b8f3cc59a34dce07bdd32555fe3c216`

## UI Coverage

The Virtual Pitch Lab now contains a collapsible **Wicket Landmark Evidence** section. It reuses the page analysis ID and setup preset and provides:

- editable analysis ID;
- load, run, and clear evidence actions;
- native ROI, temporal consensus, raw-line, accepted-axis, rejected-axis, endpoint, uncertainty, and optional scene-line view toggles;
- nullable near/far evidence summaries;
- evidence status, grade, native dimensions, supporting-frame count, accepted-axis count, top/base count, confidence, uncertainty, independent-constraint count, alignment quality, median normalized alignment residual, warnings, and extraction timing;
- legacy coarse-box versus improved-landmark registration comparison through the existing solver path;
- shape/text legend in addition to colour;
- explicit unavailable states instead of synthesized landmarks or local file paths.

The improved comparison result replaces only the lab's existing automatic camera result. It does not accept calibration, unlock metrics, close Advanced Calibration, or mount another Three.js canvas.

## API Safety

The UI uses the approved endpoints:

- `POST /video-analysis/{analysis_id}/wicket-landmark-evidence/run`
- `GET /video-analysis/{analysis_id}/wicket-landmark-evidence`
- `POST /video-analysis/{analysis_id}/wicket-landmark-evidence/clear`

Improved registration is requested by running landmark evidence with `rerun_auto_registration=true`, then loading the result from the existing auto-registration endpoint. Legacy comparison uses the normal auto-registration request, so both modes use the same solver implementation.

Debug media is accepted only from the analysis-owned evidence route or the exact `/static/video-analysis/<analysis_id>/calibration/wicket_landmarks_v1/<filename>` namespace. Windows paths, raw-video URLs, parent traversal, cross-origin URLs, protocol-relative URLs, and unrelated API paths are rejected.

The integrated backend exposes native ROI, temporal-consensus, and accepted-evidence overlays when `write_debug_media=true`. The UI does not derive public URLs from backend filesystem paths. Raw/rejected candidate media remains unavailable and is labelled accordingly.

## Contract Checks

- `near_wicket`, `far_wicket`, scene evidence, and all debug media are nullable/tolerant.
- `INSUFFICIENT_WICKETS`, `INSUFFICIENT_EVIDENCE`, partial, failed, and unknown future statuses render without assuming success.
- unavailable point/line coordinates remain nullable.
- temporal alignment is labelled **Median normalized residual** and is not presented as pixels.
- Advanced Calibration remains available in the assistance state.

## Automated Checks

From `apps/web`:

| Check | Result |
|---|---|
| `node --experimental-strip-types lib/wicketLandmarkEvidence.test.ts` | PASS |
| `tsc --noEmit --incremental false` | PASS |
| focused direct ESLint, no cache | PASS, zero errors/warnings |

The focused URL test covers valid analysis-owned relative/absolute media and rejects null, empty, Windows filesystem, parent traversal, encoded traversal, cross-origin, protocol-relative, and non-analysis paths.

The standard incremental TypeScript and `next lint` wrappers could not update pre-existing generated cache files (`tsconfig.tsbuildinfo` and `.next/cache/eslint`) because of Windows `EPERM`. Equivalent no-write/no-cache checks passed. No generated cache was committed or modified intentionally.

## Visual QA

Page: `http://127.0.0.1:3000/virtual-pitch-lab`

### Desktop - 1440 x 900

- horizontal overflow: none (`scrollWidth 1425`, `innerWidth 1440`);
- evidence section visible and expanded;
- analysis control visible;
- five default diagnostic view toggles checked;
- strongest saved evidence loaded with 10 analysis-owned debug images and 0 broken images;
- canvas count: exactly 1;
- browser console errors: 0.

### Mobile - 390 x 844

- horizontal overflow: none (`scrollWidth 375`, `innerWidth 390`);
- evidence analysis control visible and usable;
- Load, Run extraction, and Clear controls visible;
- evidence section remains expanded/collapsible;
- strongest saved evidence remained visible with 10 images and 0 broken images;
- canvas count: exactly 1;
- browser console errors: 0.

The empty state reports `No saved landmark evidence exists for this analysis.` without synthesizing content. The integrated strongest report exposes both wicket ROIs and consensus/accepted overlays through validated analysis-owned URLs.

## Integrated numerical QA

- strongest evidence: `READY`, but each wicket grade is `PARTIAL`;
- near/far axes: `3 / 3`; endpoints and transverse lines: unavailable;
- improved solver: `NEEDS_ASSISTANCE / VISUAL_ONLY`;
- improved near/far IoU: `0.41009 / 0.13372`;
- temporal stability: `0.13521`; ambiguity: `0.99999978`;
- weak analyses: three `INSUFFICIENT_EVIDENCE`, with auto registration remaining `INSUFFICIENT_WICKETS`;
- detector reuse: true for all four evaluations;
- production accepted: false; metrics unlocked: none.

The improved result is intentionally not presented as a calibration improvement. Shaft axes without supported top/base or scene constraints remain scale-ambiguous. Full backend, frontend, build, and final browser checks are recorded in the milestone close-out report.
