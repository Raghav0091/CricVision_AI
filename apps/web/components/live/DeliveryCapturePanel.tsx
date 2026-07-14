import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";


export function DeliveryCapturePanel({ deliveryCount, recording = false }: { deliveryCount: number; recording?: boolean }) {
  return (
    <Card>
      <div className="flex items-center justify-between">
        <StatusBadge label={recording ? "Recording" : "Waiting"} tone={recording ? "warn" : "good"} />
        <span className="text-3xl font-black tabular-nums">{deliveryCount}</span>
      </div>
      <h3 className="mt-6 text-lg font-bold">{recording ? "Capturing delivery" : "Waiting for delivery"}</h3>
      <p className="mt-2 text-sm leading-6 text-white/50">Rolling frame buffer integration arrives with Milestone 5.</p>
      <div className="mt-5 rounded-xl border border-dashed border-white/15 p-4 text-xs text-white/40">Last saved clip: none</div>
    </Card>
  );
}
