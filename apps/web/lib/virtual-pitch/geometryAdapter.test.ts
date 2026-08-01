import { adaptVirtualPitchResponse, VirtualPitchContractError } from "./geometryAdapter";
import { polygonIsFinite, renderableCounts } from "./geometry";
import { validVirtualPitchResponse } from "./testFixture.test-support";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const response = validVirtualPitchResponse();
const model = adaptVirtualPitchResponse(response);
assert(model.modelVersion === "v1", "Model version was not preserved.");
assert(model.landmarks[0].semanticId === "bowler_wicket_center_base", "Semantic landmark ID changed.");
assert(model.stumps[0].primitiveId === "bowler_left_stump", "Primitive ID changed.");
assert(model.dimensions.pitchLengthM === 20, "Backend pitch dimensions were not preserved.");
assert(model.dimensions.pitchWidthM === 2, "Backend pitch dimensions were not preserved.");
assert(model.polygons.every((polygon) => polygonIsFinite(polygon.vertices)), "Adapter emitted invalid polygon coordinates.");

const counts = renderableCounts(model);
assert(counts.stumps === 6, "Six stumps are required.");
assert(counts.bails === 4, "Four bails are required.");
assert(counts.lines === 5, "Line count changed during adaptation.");
assert(counts.polygons === 2, "Polygon count changed during adaptation.");

const missingStump = validVirtualPitchResponse();
(missingStump.stumps as unknown[]).pop();
let rejected = false;
try {
  adaptVirtualPitchResponse(missingStump);
} catch (error) {
  rejected = error instanceof VirtualPitchContractError && error.path === "$.stumps";
}
assert(rejected, "Incomplete wicket geometry was accepted.");

const missingCorridor = validVirtualPitchResponse();
missingCorridor.polygons = (missingCorridor.polygons as Array<{ polygon_category: string }>).filter(
  (polygon) => polygon.polygon_category !== "lbw_corridor"
);
rejected = false;
try {
  adaptVirtualPitchResponse(missingCorridor);
} catch (error) {
  rejected = error instanceof VirtualPitchContractError && error.path === "$.polygons";
}
assert(rejected, "Missing analysis corridor was accepted.");

const invalidPoint = validVirtualPitchResponse();
((invalidPoint.landmarks as Array<{ point: { x: number } }>)[0].point).x = Number.NaN;
rejected = false;
try {
  adaptVirtualPitchResponse(invalidPoint);
} catch (error) {
  rejected = error instanceof VirtualPitchContractError && error.path.endsWith(".point.x");
}
assert(rejected, "Non-finite geometry was accepted.");
