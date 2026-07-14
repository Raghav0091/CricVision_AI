import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07110d",
        panel: "#0d1c16",
        lime: "#b7f34b",
        signal: "#ff554f"
      },
      boxShadow: {
        glow: "0 0 40px rgba(183, 243, 75, 0.14)"
      }
    }
  },
  plugins: []
};

export default config;
