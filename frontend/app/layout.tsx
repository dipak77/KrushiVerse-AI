import './globals.css';
import type { Metadata } from 'next';
export const metadata: Metadata = { title: 'KrushiVerse — AI Krushi Mitra', description: 'The intelligent operating system for every farm.' };
export default function RootLayout({ children }: Readonly<{children: React.ReactNode}>) { return <html lang="en"><body>{children}</body></html>; }
