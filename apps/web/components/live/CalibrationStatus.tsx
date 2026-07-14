import { StatusBadge } from "@/components/ui/StatusBadge";


type Status = "Searching" | "Solving" | "Setup Complete" | "Failed";


export function CalibrationStatus({ status, message }: { status: Status; message?: string }) {
  const tone = status === "Setup Complete" ? "good" : status === "Failed" ? "warn" : "neutral";
  return (
    <div>
      <StatusBadge label={status} tone={tone} />
      {message && <p className="mt-3 text-sm leading-6 text-white/60">{message}</p>}
    </div>
  );
}
