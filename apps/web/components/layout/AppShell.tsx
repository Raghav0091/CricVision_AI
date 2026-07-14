import type { PropsWithChildren } from "react";

import { Header } from "./Header";
import { Sidebar } from "./Sidebar";


export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="min-h-screen">
      <Header />
      <div className="flex min-h-[calc(100vh-4rem)]">
        <Sidebar />
        <main className="min-w-0 flex-1 p-4 sm:p-6 lg:p-8">{children}</main>
      </div>
    </div>
  );
}
