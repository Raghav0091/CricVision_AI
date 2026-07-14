import type { Metadata } from "next";

import { AppShell } from "@/components/layout/AppShell";
import "./globals.css";


export const metadata: Metadata = {
  title: "CricVision Pro",
  description: "Professional live cricket delivery analysis"
};


export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body><AppShell>{children}</AppShell></body>
    </html>
  );
}
