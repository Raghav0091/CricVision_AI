"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  initialiseWicketCameraPose,
  solveWicketCameraPose,
  type CameraIntrinsics,
  type CricketPitchGeometry,
  type VideoAnalysisPreparedResponse,
  type WicketCameraPoseInitialiseResponse,
  type WicketCameraPoseResult,
  type WicketLandmarkVisibility,
  type WicketPoseLandmarkInput
} from "@/lib/api";

import { CalibrationV2LandmarkEditor } from "./CalibrationV2LandmarkEditor";


type EditorState = {
  referenceFrameUrl: string;
  imageWidth: number;
  imageHeight: number;
  pitchGeometry: CricketPitchGeometry;
  landmarks: WicketPoseLandmarkInput[];
  cameraIntrinsics: CameraIntrinsics;
  warnings: string[];
};


function stateFromInitialised(
  initialised: WicketCameraPoseInitialiseResponse
): EditorState {
  return {
    referenceFrameUrl: initialised.reference_frame_url,
    imageWidth: initialised.image_width,
    imageHeight: initialised.image_height,
    pitchGeometry: initialised.pitch_geometry,
    landmarks: initialised.landmarks,
    cameraIntrinsics: initialised.camera_intrinsics,
    warnings: initialised.warnings
  };
}


function stateFromSaved(saved: WicketCameraPoseResult): EditorState {
  return {
    referenceFrameUrl: saved.reference_frame_url,
    imageWidth: saved.image_width,
    imageHeight: saved.image_height,
    pitchGeometry: saved.pitch_geometry,
    landmarks: saved.landmarks,
    cameraIntrinsics: saved.camera_intrinsics,
    warnings: saved.camera_pose.warnings
  };
}


function changedSource(
  source: WicketPoseLandmarkInput["source"]
): WicketPoseLandmarkInput["source"] {
  return source === "manual" ? "manual" : "manually_adjusted";
}


function counterpartId(
  landmark: WicketPoseLandmarkInput,
  mode: "end" | "side"
): string {
  if (mode === "end") {
    return landmark.id.replace(
      /^(bowler|striker)_/,
      landmark.wicket_end === "bowler" ? "striker_" : "bowler_"
    );
  }
  if (landmark.stump_position === "left") {
    return landmark.id.replace("_left_", "_right_");
  }
  if (landmark.stump_position === "right") {
    return landmark.id.replace("_right_", "_left_");
  }
  return landmark.id;
}


export function CalibrationV2BPanel({
  analysis,
  initialCameraPose,
  onSolved
}: {
  analysis: VideoAnalysisPreparedResponse;
  initialCameraPose: WicketCameraPoseResult | null;
  onSolved: (result: WicketCameraPoseResult) => void;
}) {
  const [editor, setEditor] = useState<EditorState | null>(
    initialCameraPose ? stateFromSaved(initialCameraPose) : null
  );
  const [initialGuesses, setInitialGuesses] = useState<EditorState | null>(null);
  const [saved, setSaved] = useState<WicketCameraPoseResult | null>(
    initialCameraPose
  );
  const [initialising, setInitialising] = useState(false);
  const [solving, setSolving] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [semanticsConfirmed, setSemanticsConfirmed] = useState(
    initialCameraPose?.landmark_semantics_confirmed ?? false
  );
  const [userNote, setUserNote] = useState(initialCameraPose?.user_note ?? "");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (initialCameraPose) {
      setEditor(stateFromSaved(initialCameraPose));
      setSaved(initialCameraPose);
      setSemanticsConfirmed(initialCameraPose.landmark_semantics_confirmed);
      setUserNote(initialCameraPose.user_note ?? "");
      return;
    }
    void initialise();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis.analysis_id, initialCameraPose]);

  async function initialise() {
    setInitialising(true);
    setError(null);
    try {
      const result = await initialiseWicketCameraPose(analysis.analysis_id);
      const next = stateFromInitialised(result);
      setEditor(next);
      setInitialGuesses(next);
      setSaved(null);
      setSemanticsConfirmed(false);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Camera-pose landmarks could not be initialised."
      );
    } finally {
      setInitialising(false);
    }
  }

  function updateLandmark(
    landmarkId: string,
    normalizedX: number,
    normalizedY: number
  ) {
    setEditor((current) => current ? {
      ...current,
      landmarks: current.landmarks.map((landmark) => (
        landmark.id === landmarkId
          ? {
              ...landmark,
              normalized_x: normalizedX,
              normalized_y: normalizedY,
              source: changedSource(landmark.source),
              confidence: null,
              visibility: "visible"
            }
          : landmark
      ))
    } : current);
    setSaved(null);
    setSemanticsConfirmed(false);
  }

  function changeVisibility(
    landmarkId: string,
    visibility: WicketLandmarkVisibility
  ) {
    setEditor((current) => current ? {
      ...current,
      landmarks: current.landmarks.map((landmark) => (
        landmark.id === landmarkId ? { ...landmark, visibility } : landmark
      ))
    } : current);
    setSaved(null);
    setSemanticsConfirmed(false);
  }

  function swap(mode: "end" | "side") {
    setEditor((current) => {
      if (!current) return current;
      const byId = new Map(
        current.landmarks.map((landmark) => [landmark.id, landmark])
      );
      return {
        ...current,
        landmarks: current.landmarks.map((landmark) => {
          const other = byId.get(counterpartId(landmark, mode));
          if (!other || other.id === landmark.id) return landmark;
          return {
            ...landmark,
            normalized_x: other.normalized_x,
            normalized_y: other.normalized_y,
            source: changedSource(other.source),
            confidence: null,
            visibility: other.visibility
          };
        })
      };
    });
    setSaved(null);
    setSemanticsConfirmed(false);
  }

  function resetGuesses() {
    if (!initialGuesses) {
      void initialise();
      return;
    }
    setEditor({
      ...initialGuesses,
      landmarks: initialGuesses.landmarks.map((landmark) => ({ ...landmark }))
    });
    setSaved(null);
    setSemanticsConfirmed(false);
  }

  async function solvePose() {
    if (!editor) return;
    setSolving(true);
    setError(null);
    try {
      const result = await solveWicketCameraPose(analysis.analysis_id, {
        analysis_id: analysis.analysis_id,
        landmarks: editor.landmarks,
        pitch_geometry: editor.pitchGeometry,
        camera_intrinsics: editor.cameraIntrinsics,
        landmark_semantics_confirmed: semanticsConfirmed,
        user_note: userNote.trim() || null
      });
      setSaved(result);
      setEditor(stateFromSaved(result));
      onSolved(result);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Wicket-based camera pose could not be solved."
      );
    } finally {
      setSolving(false);
    }
  }

  const usableLandmarks = editor?.landmarks.filter(
    (landmark) => (
      landmark.visibility === "visible"
      || landmark.visibility === "uncertain"
    )
  ) ?? [];
  const statusTone = saved?.status === "ready" || saved?.status === "usable"
    ? "good"
    : saved
      ? "warn"
      : "neutral";

  return (
    <section className="mt-8 border-t border-white/10 pt-8">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <StatusBadge
            label={saved ? `Camera Pose ${saved.status.replaceAll("_", " ")}` : "Calibration v2B"}
            tone={statusTone}
          />
          <h2 className="mt-4 text-2xl font-black">
            Wicket-based camera pose
          </h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-white/55">
            Use the two visible wickets as a known 3D target. Base means the
            ground-contact point. Top means the top of the stump body and
            explicitly excludes bails.
          </p>
        </div>
        <Button
          variant="secondary"
          disabled={initialising || solving}
          onClick={() => void initialise()}
        >
          {initialising ? "Initialising..." : "Reinitialise 12 Markers"}
        </Button>
      </div>

      {error && (
        <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}

      <ol className="mt-5 grid gap-2 text-xs leading-5 text-white/55 md:grid-cols-5">
        {[
          "Place BASE markers at ground contact.",
          "Place TOP markers at the top of each stump body.",
          "Do not place TOP markers on bails.",
          "Mark unclear points unavailable.",
          "Review both ends and solve."
        ].map((instruction, index) => (
          <li
            key={instruction}
            className="rounded-xl border border-white/10 bg-black/20 p-3"
          >
            <span className="mr-2 font-black text-lime">{index + 1}.</span>
            {instruction}
          </li>
        ))}
      </ol>

      {editor && (
        <>
          <div className="mt-6 flex justify-end">
            <button
              type="button"
              className="text-xs font-bold text-lime underline"
              onClick={() => setShowLabels((current) => !current)}
            >
              {showLabels ? "Hide labels" : "Show labels"}
            </button>
          </div>
          <div className="mt-3">
            <CalibrationV2LandmarkEditor
              imageUrl={editor.referenceFrameUrl}
              imageWidth={editor.imageWidth}
              imageHeight={editor.imageHeight}
              landmarks={editor.landmarks}
              disabled={solving}
              showLabels={showLabels}
              onLandmarkChange={updateLandmark}
            />
          </div>

          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            {(["striker", "bowler"] as const).map((wicketEnd) => (
              <div
                key={wicketEnd}
                className="rounded-xl border border-white/10 bg-black/20 p-4"
              >
                <p className="font-black uppercase">{wicketEnd} wicket</p>
                <div className="mt-3 grid gap-2">
                  {(["top", "base"] as const).map((pointType) => (
                    <div key={pointType}>
                      <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
                        {pointType}
                      </p>
                      <div className="mt-2 grid grid-cols-3 gap-2">
                        {editor.landmarks
                          .filter((landmark) => (
                            landmark.wicket_end === wicketEnd
                            && landmark.point_type === pointType
                          ))
                          .map((landmark) => (
                            <label
                              key={landmark.id}
                              className="rounded-lg border border-white/10 p-2 text-xs"
                            >
                              <span className="font-black">
                                {pointType === "top" ? "T" : "B"}-
                                {landmark.stump_position[0].toUpperCase()}
                              </span>
                              <select
                                className="mt-2 w-full rounded border border-white/10 bg-[#101418] p-1 text-[11px]"
                                value={landmark.visibility}
                                disabled={solving}
                                onChange={(event) => changeVisibility(
                                  landmark.id,
                                  event.target.value as WicketLandmarkVisibility
                                )}
                              >
                                <option value="visible">Visible</option>
                                <option value="uncertain">Uncertain</option>
                                <option value="occluded">Occluded</option>
                                <option value="unavailable">Unavailable</option>
                              </select>
                            </label>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-3">
            <Button variant="secondary" disabled={solving} onClick={resetGuesses}>
              Reset Initial Guesses
            </Button>
            <Button variant="secondary" disabled={solving} onClick={() => swap("end")}>
              Swap Wicket Ends
            </Button>
            <Button variant="secondary" disabled={solving} onClick={() => swap("side")}>
              Swap Left / Right
            </Button>
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <div className="rounded-xl border border-white/10 bg-black/20 p-4">
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-white/40">
                Known 3D wicket model
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-white/55">
                <span>Pitch length</span><strong>{editor.pitchGeometry.pitch_length_m} m</strong>
                <span>Wicket outside width</span><strong>{editor.pitchGeometry.wicket_width_m} m</strong>
                <span>Stump diameter</span><strong>{editor.pitchGeometry.stump_diameter_m} m</strong>
                <span>Stump body height</span><strong>{editor.pitchGeometry.wicket_height_m} m</strong>
              </div>
            </div>
            <div className="rounded-xl border border-[#ffca68]/25 bg-[#ffca68]/[0.05] p-4">
              <p className="text-xs font-bold uppercase tracking-[0.12em] text-[#ffe0a3]">
                Camera intrinsics: {editor.cameraIntrinsics.source.replaceAll("_", " ")}
              </p>
              <p className="mt-2 text-sm text-white/60">
                fx {editor.cameraIntrinsics.fx.toFixed(2)} · fy {editor.cameraIntrinsics.fy.toFixed(2)}
                {" "}· principal point ({editor.cameraIntrinsics.cx.toFixed(1)}, {editor.cameraIntrinsics.cy.toFixed(1)})
              </p>
              <p className="mt-2 text-xs leading-5 text-white/40">
                Distortion: {editor.cameraIntrinsics.distortion_model_source.replaceAll("_", " ")}.
                Estimated intrinsics reduce pose confidence and are not a
                calibrated device profile.
              </p>
            </div>
          </div>

          {editor.warnings.length > 0 && !saved && (
            <div className="mt-4 rounded-xl border border-[#ffca68]/20 bg-black/20 p-3 text-xs leading-5 text-[#ffe0a3]">
              {editor.warnings.map((warning) => <p key={warning}>• {warning}</p>)}
            </div>
          )}

          <label className="mt-5 flex items-start gap-3 rounded-xl border border-white/10 bg-black/20 p-4 text-sm leading-6">
            <input
              className="mt-1 h-4 w-4 accent-[#d5ff6b]"
              type="checkbox"
              checked={semanticsConfirmed}
              disabled={solving}
              onChange={(event) => {
                setSemanticsConfirmed(event.target.checked);
                setSaved(null);
              }}
            />
            <span>
              I confirm every available BASE is a stump-ground contact and
              every available TOP is the top of the stump body, excluding bails.
            </span>
          </label>

          <label htmlFor="camera-pose-note" className="mt-5 block text-sm font-bold">
            Camera-pose note <span className="font-normal text-white/35">(optional)</span>
          </label>
          <textarea
            id="camera-pose-note"
            className="mt-2 min-h-20 w-full rounded-xl border border-white/10 bg-black/25 p-3 text-sm outline-none focus:border-lime/40"
            maxLength={1000}
            value={userNote}
            disabled={solving}
            onChange={(event) => {
              setUserNote(event.target.value);
              setSaved(null);
            }}
          />

          <Button
            className="mt-5"
            disabled={solving || !semanticsConfirmed}
            onClick={() => void solvePose()}
          >
            {solving ? "Solving Camera Pose..." : "Solve Wicket-Based Camera Pose"}
          </Button>
          <span className="ml-3 text-xs text-white/40">
            {usableLandmarks.length}/12 usable landmarks
          </span>
        </>
      )}

      {saved && (
        <div className="mt-8 rounded-xl border border-white/10 bg-white/[0.03] p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <StatusBadge
                label={saved.status.replaceAll("_", " ")}
                tone={statusTone}
              />
              <p className="mt-3 text-sm text-white/55">{saved.message}</p>
            </div>
            <a
              className="text-xs font-bold text-lime underline"
              href={saved.camera_pose_url}
              target="_blank"
              rel="noreferrer"
            >
              Open camera_pose.json
            </a>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Metric label="Landmarks used" value={`${saved.camera_pose.landmark_count}/12`} />
            <Metric label="RANSAC inliers" value={`${saved.camera_pose.ransac_inlier_ids.length}/${saved.camera_pose.landmark_count}`} />
            <Metric label="Reprojection RMSE" value={saved.camera_pose.reprojection_rmse_px != null ? `${saved.camera_pose.reprojection_rmse_px.toFixed(2)} px` : "Unavailable"} />
            <Metric label="Max residual" value={saved.camera_pose.reprojection_max_px != null ? `${saved.camera_pose.reprojection_max_px.toFixed(2)} px` : "Unavailable"} />
            <Metric label="Solver" value={saved.camera_pose.solver_method} />
            <Metric label="Refinement" value={saved.camera_pose.refinement_method ?? "Not used"} />
            <Metric label="Intrinsics" value={saved.camera_intrinsics.source.replaceAll("_", " ")} />
            <Metric label="Overall quality" value={`${(saved.quality.overall_pose_quality * 100).toFixed(1)}%`} />
            <Metric label="Camera height" value={saved.camera_pose.camera_height_m != null ? `${saved.camera_pose.camera_height_m.toFixed(2)} m` : "Unavailable"} />
            <Metric label="Positive depth" value={saved.camera_pose.positive_depth_for_all_used_landmarks == null ? "Unavailable" : saved.camera_pose.positive_depth_for_all_used_landmarks ? "Yes" : "No"} />
            <Metric label="Faces pitch" value={saved.camera_pose.camera_faces_pitch == null ? "Unavailable" : saved.camera_pose.camera_faces_pitch ? "Yes" : "No"} />
            <Metric label="Pose accepted" value={saved.camera_pose.accepted ? "Yes" : "No"} />
          </div>

          {saved.camera_pose.camera_position_world && (
            <p className="mt-4 rounded-lg border border-white/10 bg-black/20 p-3 font-mono text-xs text-white/55">
              Camera world position: [{saved.camera_pose.camera_position_world.map(
                (value) => value.toFixed(4)
              ).join(", ")}] m
            </p>
          )}

          {(saved.camera_pose.warnings.length > 0 || saved.camera_pose.rejection_reasons.length > 0) && (
            <div className="mt-4 rounded-lg border border-[#ffca68]/20 bg-black/20 p-3 text-xs leading-5 text-[#ffe0a3]">
              {saved.camera_pose.warnings.map((warning) => <p key={warning}>• {warning}</p>)}
              {saved.camera_pose.rejection_reasons.map((reason) => (
                <p key={reason}>• Rejected: {reason.replaceAll("_", " ")}</p>
              ))}
            </div>
          )}

          {saved.camera_pose.reprojection_diagnostics.length > 0 && (
            <div className="mt-5 overflow-x-auto">
              <table className="w-full min-w-[680px] text-left text-xs">
                <thead className="text-white/35">
                  <tr>
                    <th className="p-2">Landmark</th>
                    <th className="p-2">Observed</th>
                    <th className="p-2">Projected</th>
                    <th className="p-2">Residual</th>
                    <th className="p-2">RANSAC</th>
                  </tr>
                </thead>
                <tbody>
                  {saved.camera_pose.reprojection_diagnostics.map((diagnostic) => (
                    <tr key={diagnostic.landmark_id} className="border-t border-white/10">
                      <td className="p-2 font-mono">{diagnostic.landmark_id}</td>
                      <td className="p-2">({diagnostic.observed_pixel_x.toFixed(1)}, {diagnostic.observed_pixel_y.toFixed(1)})</td>
                      <td className="p-2">({diagnostic.projected_pixel_x.toFixed(1)}, {diagnostic.projected_pixel_y.toFixed(1)})</td>
                      <td className="p-2">{diagnostic.residual_px.toFixed(2)} px</td>
                      <td className="p-2">{diagnostic.ransac_inlier ? "Inlier" : "Outlier"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="mt-5 h-auto w-full rounded-lg bg-black object-contain"
            src={`${saved.camera_pose_overlay_url}?v=${encodeURIComponent(saved.updated_at)}`}
            alt="Authoritative wicket-based camera-pose reprojection overlay"
          />
        </div>
      )}
    </section>
  );
}


function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
      <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
        {label}
      </p>
      <p className="mt-2 break-words text-sm font-black capitalize">{value}</p>
    </div>
  );
}
