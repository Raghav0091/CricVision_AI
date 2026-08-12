import type { ButtonHTMLAttributes, PropsWithChildren } from "react";


type ButtonProps = PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>> & {
  variant?: "primary" | "secondary" | "danger" | "camera";
};


export function Button({ children, className = "", variant = "primary", ...props }: ButtonProps) {
  const styles = {
    primary: "bg-lime text-ink hover:bg-[#c7ff65]",
    secondary: "border border-white/15 bg-white/5 text-white hover:bg-white/10",
    danger: "border border-signal/40 bg-signal/10 text-[#ffaaa6] hover:bg-signal/20",
    // Camera stages are light-on-dark over a live video frame, where the lime
    // primary reads as part of the scene rather than as chrome.
    camera: "bg-white text-[#1A73E8] uppercase tracking-wide hover:bg-white/90"
  };
  const shape = variant === "camera" ? "rounded-full px-8 py-3" : "rounded-xl px-5 py-3";
  return (
    <button
      className={`${shape} text-sm font-bold transition disabled:cursor-not-allowed disabled:opacity-45 ${styles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
