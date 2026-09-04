import type { NextConfig } from "next";

// Backend origin for CSP connect-src. Override with BACKEND_ORIGIN at build
// time so preview environments and alternate domains aren't blocked.
const backendOrigin = process.env.BACKEND_ORIGIN || "https://inditrade-backend.onrender.com";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
          { key: 'X-XSS-Protection', value: '1; mode=block' },
          { key: 'Content-Security-Policy', value: `default-src 'self'; script-src 'self' 'unsafe-inline' https://unpkg.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self' ${backendOrigin} http://localhost:8000;` },
        ],
      },
    ];
  },
};

export default nextConfig;
