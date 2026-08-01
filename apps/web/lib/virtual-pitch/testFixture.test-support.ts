type Point = { x: number; y: number; z: number };

const point = (x: number, y: number, z = 0): Point => ({ x, y, z });

export function validVirtualPitchResponse(): Record<string, unknown> {
  const stumps = (["bowler", "striker"] as const).flatMap((end, endIndex) =>
    (["left", "middle", "right"] as const).map((stumpIndex, stumpIndexNumber) => ({
      primitive_id: `${end}_${stumpIndex}_stump`,
      centre: point(stumpIndexNumber - 1, endIndex * 20, 0.5),
      radius_m: 0.04,
      height_m: 1,
      orientation: point(0, 0, 1),
      end,
      stump_index: stumpIndex,
      geometry_class: "official"
    }))
  );
  const bails = (["bowler", "striker"] as const).flatMap((end, endIndex) =>
    (["left_middle", "middle_right"] as const).map((bailIndex, bailIndexNumber) => ({
      primitive_id: `${end}_${bailIndex}_bail`,
      start: point(bailIndexNumber - 1, endIndex * 20, 1),
      end_point: point(bailIndexNumber, endIndex * 20, 1),
      radius_m: 0.02,
      end,
      bail_index: bailIndex,
      geometry_class: "official",
      cosmetic: true
    }))
  );
  const categories = ["pitch_boundary", "bowling_crease", "popping_crease", "return_crease", "centreline"];
  return {
    virtual_pitch_model_version: "v1",
    coordinate_system: {
      units: "metres",
      handedness: "right_handed",
      origin: "bowler_end_middle_stump_base",
      x_axis: "lateral_camera_neutral_right",
      y_axis: "bowler_to_striker",
      z_axis: "up",
      description: "Test coordinate system",
      off_leg_assignment: "not_assigned"
    },
    dimensions: {
      pitch_length_m: 20,
      pitch_width_m: 2,
      wicket_width_m: 1,
      stump_height_m: 1,
      stump_diameter_min_m: 0.07,
      stump_diameter_max_m: 0.08,
      bowling_crease_length_m: 3,
      popping_crease_offset_m: 1,
      return_crease_offset_m: 1.5
    },
    landmarks: [{
      semantic_id: "bowler_wicket_center_base",
      point: point(0, 0),
      geometry_category: "wicket",
      geometry_class: "official",
      end: "bowler",
      calibration_anchor: true,
      description: "Test landmark"
    }],
    stumps,
    bails,
    line_segments: categories.map((line_category, index) => ({
      primitive_id: `line_${line_category}`,
      start: point(-1, index * 2),
      end_point: point(1, index * 2),
      line_category,
      geometry_class: line_category === "centreline" ? "analytical" : "official",
      line_width_m: 0.05,
      end: "both",
      profile_id: line_category === "centreline" ? "analytical" : null
    })),
    polygons: [
      {
        primitive_id: "pitch_surface",
        vertices: [point(-1, 0), point(1, 0), point(1, 20), point(-1, 20)],
        polygon_category: "pitch_surface",
        geometry_class: "official",
        end: "both",
        profile_id: null,
        display_opacity: 0.1
      },
      {
        primitive_id: "centre_corridor",
        vertices: [point(-0.5, 0), point(0.5, 0), point(0.5, 20), point(-0.5, 20)],
        polygon_category: "lbw_corridor",
        geometry_class: "analytical",
        end: "both",
        profile_id: "analytical",
        display_opacity: 0.2
      }
    ],
    profiles: [{
      profile_id: "analytical",
      label: "Analytical",
      geometry_class: "analytical",
      description: "Test profile",
      enabled_primitive_ids: ["line_centreline", "centre_corridor"],
      universal_official_geometry: false
    }],
    display_rounding: {
      stored_precision: "full_float",
      display_decimal_places: 3,
      display_units: "metres"
    },
    synthetic_camera_names: ["test_camera"]
  };
}
