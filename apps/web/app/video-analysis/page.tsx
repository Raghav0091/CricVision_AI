import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";


export default function VideoAnalysisPage() {
  return (
    <div className="mx-auto max-w-4xl py-6">
      <StatusBadge label="Legacy workflow" />
      <h1 className="mt-5 text-4xl font-black tracking-tight">Video Analysis</h1>
      <Card className="mt-8">
        <p className="text-white/60">Uploaded clip analysis remains in the Streamlit prototype while the worker API is connected.</p>
      </Card>
    </div>
  );
}
