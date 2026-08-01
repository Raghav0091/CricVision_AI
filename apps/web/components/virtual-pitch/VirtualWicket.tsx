import type { BailModel, MaterialStyle, StumpModel } from "@/lib/virtual-pitch";

import { VirtualBails } from "./VirtualBails";
import { VirtualStump } from "./VirtualStump";


export function VirtualWicket({
  wicketEnd,
  stumps,
  bails,
  stumpMaterial,
  bailMaterial,
  showStumps = true,
  showBails = true,
}: {
  wicketEnd: "bowler" | "striker";
  stumps: readonly StumpModel[];
  bails: readonly BailModel[];
  stumpMaterial: MaterialStyle;
  bailMaterial: MaterialStyle;
  showStumps?: boolean;
  showBails?: boolean;
}) {
  const wicketStumps = stumps.filter((stump) => stump.end === wicketEnd);
  const wicketBails = bails.filter((bail) => bail.end === wicketEnd);
  return (
    <group name={`${wicketEnd}-wicket`}>
      {showStumps && wicketStumps.map((stump) => (
        <VirtualStump key={stump.primitiveId} stump={stump} material={stumpMaterial} />
      ))}
      {showBails && (
        <VirtualBails bails={wicketBails} material={bailMaterial} />
      )}
    </group>
  );
}
