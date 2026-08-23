import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // CI and local verification can use an isolated output directory when another
  // Next process is holding the default `.next` trace file open.
  distDir: process.env.NEXT_DIST_DIR ?? ".next",
};

export default nextConfig;
