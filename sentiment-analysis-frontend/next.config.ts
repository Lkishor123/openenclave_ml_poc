/** @type {import('next').NextConfig} */
const nextConfig = {
  // This is the required change.
  // It tells Next.js to create a standalone output folder.
  output: 'standalone',
};

export default nextConfig;
