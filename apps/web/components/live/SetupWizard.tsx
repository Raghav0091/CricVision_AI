"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/Button";


const SKIP_STORAGE_KEY = "cricvision.live.skipSetupWizard";

type Example = {
  src: string;
  label: string;
  tone: "good" | "bad";
};

type SetupCard = {
  title: string;
  subtitle?: string;
  examples: Example[];
  icon?: "clock";
};

// Card copy mirrors the FullTrack AI live setup flow so bowlers who have used
// either app land on the same instructions in the same order.
const SETUP_CARDS: SetupCard[] = [
  {
    title: "You Need a Tripod and 6 Stumps",
    examples: [{ src: "/setup/tripod-and-stumps.jpg", label: "", tone: "good" }]
  },
  {
    title: "Camera Position Guidelines",
    subtitle: "4m behind non-striker stumps",
    examples: [
      { src: "/setup/position-good.jpg", label: "Good", tone: "good" },
      { src: "/setup/position-too-far-back.jpg", label: "Too Far Back", tone: "bad" }
    ]
  },
  {
    title: "Camera Height Guidelines",
    subtitle: "Higher is better",
    examples: [
      { src: "/setup/height-good.jpg", label: "Good", tone: "good" },
      { src: "/setup/height-too-low.jpg", label: "Too Low", tone: "bad" }
    ]
  },
  {
    title: "Don't Block the Camera",
    examples: [
      { src: "/setup/blocking-good.jpg", label: "Good", tone: "good" },
      { src: "/setup/blocking-obstructing.jpg", label: "Obstructing", tone: "bad" }
    ]
  },
  {
    title: "You can move the camera to the side to prevent blocking",
    examples: [{ src: "/setup/side-camera-good.jpg", label: "Good", tone: "good" }]
  },
  {
    title: "Ball tracking is available 3-5 minutes after upload",
    subtitle: "Bowl a few balls before checking",
    examples: [],
    icon: "clock"
  }
];

export const SETUP_CARD_COUNT = SETUP_CARDS.length;


export function shouldSkipSetupWizard(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(SKIP_STORAGE_KEY) === "true";
}


function ClockFace() {
  return (
    <svg viewBox="0 0 120 120" role="img" aria-label="Clock" className="h-40 w-40">
      <circle cx="60" cy="60" r="54" fill="#f5f7fa" stroke="#0d1c16" strokeWidth="6" />
      {Array.from({ length: 12 }, (_, index) => {
        const angle = (index * Math.PI) / 6;
        const inner = index % 3 === 0 ? 38 : 43;
        return (
          <line
            key={index}
            x1={60 + inner * Math.sin(angle)}
            y1={60 - inner * Math.cos(angle)}
            x2={60 + 48 * Math.sin(angle)}
            y2={60 - 48 * Math.cos(angle)}
            stroke="#0d1c16"
            strokeWidth={index % 3 === 0 ? 5 : 3}
            strokeLinecap="round"
          />
        );
      })}
      <line x1="60" y1="60" x2="60" y2="32" stroke="#0d1c16" strokeWidth="6" strokeLinecap="round" />
      <line x1="60" y1="60" x2="80" y2="72" stroke="#0d1c16" strokeWidth="5" strokeLinecap="round" />
    </svg>
  );
}


function ExampleFrame({ example }: { example: Example }) {
  const [failed, setFailed] = useState(false);
  const good = example.tone === "good";
  const border = good ? "border-[#26d867]" : "border-signal";
  const text = good ? "text-[#26d867]" : "text-signal";

  return (
    <figure className="flex flex-col items-center gap-2">
      <div className={`relative aspect-[3/4] w-full max-w-[210px] overflow-hidden rounded-lg border-4 ${border} bg-black/60`}>
        {failed ? (
          // ponytail: reference photos are supplied by the operator; until they are
          // dropped into public/setup the card still teaches the rule via its label.
          <div className="grid h-full place-items-center px-3 text-center text-[11px] leading-5 text-white/45">
            Add <span className="font-mono">{example.src}</span>
          </div>
        ) : (
          <Image
            src={example.src}
            alt={example.label || "Setup example"}
            fill
            sizes="210px"
            className="object-cover"
            onError={() => setFailed(true)}
            unoptimized
          />
        )}
      </div>
      {example.label && <figcaption className={`text-sm font-black ${text}`}>{example.label}</figcaption>}
    </figure>
  );
}


export function SetupWizard({ onComplete, onBack }: { onComplete: () => void; onBack?: () => void }) {
  const [index, setIndex] = useState(0);
  const [skipNextTime, setSkipNextTime] = useState(false);

  useEffect(() => {
    setSkipNextTime(shouldSkipSetupWizard());
  }, []);

  const card = SETUP_CARDS[index];
  const isLast = index === SETUP_CARDS.length - 1;

  function toggleSkip() {
    const next = !skipNextTime;
    setSkipNextTime(next);
    window.localStorage.setItem(SKIP_STORAGE_KEY, String(next));
  }

  function goBack() {
    if (index > 0) {
      setIndex(index - 1);
      return;
    }
    onBack?.();
  }

  function goNext() {
    if (isLast) {
      onComplete();
      return;
    }
    setIndex(index + 1);
  }

  return (
    <section className="mx-auto flex min-h-[36rem] w-full max-w-md flex-col rounded-2xl border border-white/10 bg-black p-5 shadow-glow">
      <div>
        <button
          type="button"
          onClick={goBack}
          className="rounded-lg px-2 py-1 text-sm font-bold text-white/80 transition hover:bg-white/10 hover:text-white"
        >
          &lt; Back
        </button>
      </div>

      <div className="mt-4 flex flex-1 flex-col items-center">
        <h2 className="text-center text-xl font-black leading-7 text-white">{card.title}</h2>
        {card.subtitle && <p className="mt-3 text-center text-sm font-bold text-white/80">{card.subtitle}</p>}

        <div className="mt-6 flex w-full flex-1 flex-col items-center justify-center gap-5">
          {card.icon === "clock" && <ClockFace />}
          {card.examples.map((example) => (
            <ExampleFrame key={example.src} example={example} />
          ))}
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between gap-3 border-t border-white/10 pt-4">
        <label className="flex cursor-pointer items-center gap-2 text-xs font-bold text-white/70">
          <input
            type="checkbox"
            checked={skipNextTime}
            onChange={toggleSkip}
            className="h-4 w-4 shrink-0 appearance-none rounded-full border-2 border-white/60 checked:border-lime checked:bg-lime"
          />
          Skip Next Time
        </label>
        <span className="text-sm font-black tabular-nums text-white/70">
          {index + 1}/{SETUP_CARDS.length}
        </span>
        <Button onClick={goNext}>{isLast ? "Start" : "Next"}</Button>
      </div>
    </section>
  );
}
