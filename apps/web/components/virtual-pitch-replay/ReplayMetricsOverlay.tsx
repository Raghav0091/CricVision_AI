import type { ReplayMetric } from "@/lib/virtual-pitch-replay/types";

function formatMetric(metric: ReplayMetric, digits = 1): string {
  if (metric.status !== "AVAILABLE" || metric.value === null) {
    return metric.unavailable_reason ?? "Unavailable";
  }
  return `${metric.value.toFixed(digits)} ${metric.unit}`;
}

function metricConfidence(metric: ReplayMetric): string | null {
  if (metric.confidence === null) return null;
  return metric.confidence.toFixed(2);
}

function MetricRow({
  label,
  metric
}: {
  label: string;
  metric: ReplayMetric;
}) {
  const confidence = metricConfidence(metric);
  return (
    <div className="rounded-lg bg-black/30 px-3 py-2">
      <dt className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">{label}</dt>
      <dd className="mt-1 text-sm font-bold text-white">{formatMetric(metric)}</dd>
      {confidence && metric.status === "AVAILABLE" ? (
        <p className="mt-0.5 text-[10px] text-white/40">Confidence {confidence}</p>
      ) : null}
    </div>
  );
}

export function ReplayMetricsOverlay({
  measurementValidity,
  releaseSpeed,
  averagePreBounceSpeed,
  speedAtBounce,
  deliveryLength,
  estimatedLateralDeviation
}: {
  measurementValidity: string;
  releaseSpeed: ReplayMetric;
  averagePreBounceSpeed: ReplayMetric;
  speedAtBounce: ReplayMetric;
  deliveryLength: ReplayMetric;
  estimatedLateralDeviation: ReplayMetric;
}) {
  const showMetrics = measurementValidity === "CALIBRATED";

  return (
    <aside
      className="pointer-events-none absolute right-3 top-3 z-20 w-[min(100%,18rem)] space-y-2"
      aria-label="Replay metrics"
    >
      <div className="rounded-lg border border-white/10 bg-[#050806]/85 px-3 py-2 backdrop-blur-sm">
        <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-lime">Measurement</p>
        <p className="mt-1 text-sm font-bold capitalize text-white">{measurementValidity.replaceAll("_", " ").toLowerCase()}</p>
      </div>

      {showMetrics ? (
        <dl className="grid gap-2">
          <MetricRow label="Release speed" metric={releaseSpeed} />
          <MetricRow label="Avg pre-bounce speed" metric={averagePreBounceSpeed} />
          <MetricRow label="Speed at bounce" metric={speedAtBounce} />
          <MetricRow label="Delivery length" metric={deliveryLength} />
          <MetricRow label="Estimated lateral deviation" metric={estimatedLateralDeviation} />
        </dl>
      ) : (
        <div className="rounded-lg border border-white/10 bg-[#050806]/85 px-3 py-2 text-xs leading-5 text-white/55 backdrop-blur-sm">
          Measured delivery metrics are not available for this replay mode.
        </div>
      )}
    </aside>
  );
}
