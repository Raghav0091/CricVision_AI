import type { ButtonHTMLAttributes, PropsWithChildren } from "react";


type ButtonProps = PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>> & {
  variant?: "primary" | "secondary" | "danger";
};


export function Button({ children, className = "", variant = "primary", ...props }: ButtonProps) {
  const styles = {
    primary: "bg-lime text-ink hover:bg-[#c7ff65]",
    secondary: "border border-white/15 bg-white/5 text-white hover:bg-white/10",
    danger: "border border-signal/40 bg-signal/10 text-[#ffaaa6] hover:bg-signal/20"
  };
  return (
    <button
      className={`rounded-xl px-5 py-3 text-sm font-bold transition disabled:cursor-not-allowed disabled:opacity-45 ${styles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
