import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import rosePine from "starlight-theme-rose-pine";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const site = process.env.ASTRO_SITE ?? "https://tyler-romero.github.io/greyhound";
const base = process.env.ASTRO_BASE;

export default defineConfig({
  site,
  ...(base ? { base } : {}),
  srcDir: ".",
  integrations: [
    starlight({
      title: "greyhound",
      description: "High-performance, fused GPU kernels for PyTorch training workloads",
      logo: {
        src: "./public/assets/logo.png",
        alt: "greyhound",
      },
      favicon: "/assets/favicon.ico",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/tyler-romero/greyhound",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/tyler-romero/greyhound/edit/main/docs/",
      },
      customCss: ["./styles/starlight.css"],
      sidebar: [
        {
          label: "Start Here",
          items: [
            { slug: "index", label: "Home" },
            { slug: "installation" },
            { slug: "benchmarking" },
          ],
        },
        {
          label: "Kernels",
          items: [
            { slug: "kernels/chunked_linear_cross_entropy" },
            { slug: "kernels/cross_entropy" },
            { slug: "kernels/causal_conv1d" },
            { slug: "kernels/selective_log_softmax" },
          ],
        },
        {
          label: "Strategies",
          items: [
            { slug: "strategies/chunked_linear_loss" },
          ],
        },
        {
          label: "Bonus",
          items: [
            { slug: "bonus/newton_schulz" },
          ],
        },
        {
          label: "Reference",
          items: [
            { slug: "api" },
            { slug: "api/functional" },
            { slug: "api/modules" },
          ],
        },
        {
          label: "Development",
          items: [
            { slug: "development/changelog" },
          ],
        },
      ],
      plugins: [rosePine()],
    }),
  ],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex],
    }),
  },
  vite: {
    server: {
      watch: {
        ignored: [
          "**/.venv/**",
          "**/.pytest_cache/**",
          "**/.ruff_cache/**",
          "**/dist/**",
          "**/build/**",
          "**/node_modules/**",
        ],
      },
    },
  },
});
