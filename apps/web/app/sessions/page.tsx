import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";


export default function SessionsPage() {
  return (
    <div className="mx-auto max-w-4xl py-6">
      <StatusBadge label="Foundation" />
      <h1 className="mt-5 text-4xl font-black tracking-tight">Sessions</h1>
      <Card className="mt-8">
        <p className="text-white/60">Session history will appear here after persistent storage is introduced.</p>
      </Card>
    </div>
  );
}
