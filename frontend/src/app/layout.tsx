import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "IndiTrade AI - Global Trade Analytics",
  description: "Intelligent foreign trade policy and forecasting dashboard powered by AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        {children}
      </body>
    </html>
  );
}
