export function VirtualPitchLighting({
  ambientColour,
  keyColour,
  lowPerformance = false
}: {
  ambientColour: string;
  keyColour: string;
  lowPerformance?: boolean;
}) {
  return (
    <group name="virtual-pitch-lighting">
      <ambientLight color={ambientColour} intensity={lowPerformance ? 1.15 : 0.85} />
      {!lowPerformance && (
        <hemisphereLight color={keyColour} groundColor={ambientColour} intensity={0.5} />
      )}
      <directionalLight
        color={keyColour}
        intensity={lowPerformance ? 1.2 : 1.65}
        position={[4, 9, 2]}
        castShadow={false}
      />
    </group>
  );
}
