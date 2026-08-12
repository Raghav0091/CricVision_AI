/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // ponytail: one public URL for phone testing; Next proxies API to local FastAPI.
    return [
      {
        source: "/cricvision-api/:path*",
        destination: "http://127.0.0.1:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
