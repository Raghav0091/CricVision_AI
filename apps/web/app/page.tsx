import Link from "next/link";

import { Card } from "@/components/ui/Card";


const destinations = [
  { href: "/live", eyebrow: "Live", title: "Live Session", copy: "Calibrate a fixed camera and capture deliveries." },
  { href: "/video-analysis", eyebrow: "Analysis", title: "Video Analysis", copy: "Upload a clip, calibrate the wickets, detect and track the ball, read the speed." }
];


export default function Dashboard() {
  return (
    <div className="mx-auto max-w-6xl py-6">
      <p className="text-xs font-bold uppercase tracking-[0.25em] text-lime">CricVision AI</p>
      <h1 className="mt-4 max-w-3xl text-4xl font-black tracking-[-0.04em] sm:text-6xl">See every delivery more clearly.</h1>
      <p className="mt-5 max-w-2xl text-base leading-7 text-white/55">Live capture and uploaded video analysis, measuring delivery speed from a single camera.</p>
      <div className="mt-10 grid gap-4 md:grid-cols-2">
        {destinations.map((item) => (
          <Link key={item.href} href={item.href} className="group">
            <Card className="h-full transition group-hover:-translate-y-1 group-hover:border-lime/30">
              <p className="text-xs font-bold uppercase tracking-[0.16em] text-lime">{item.eyebrow}</p>
              <h2 className="mt-8 text-xl font-bold">{item.title}</h2>
              <p className="mt-2 text-sm leading-6 text-white/50">{item.copy}</p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
