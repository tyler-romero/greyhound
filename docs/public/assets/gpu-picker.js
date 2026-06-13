const STORAGE_KEY = "greyhound-benchmark-gpu";

const GPU_POWER_ORDER = [
  /b200/i,
  /h100/i,
  /a100/i,
  /rtx\s*6000/i,
  /rtx\s*5090/i,
  /rtx\s*4090/i,
  /rtx\s*3090/i,
];

function getGpuPowerRank(gpu) {
  const rank = GPU_POWER_ORDER.findIndex((pattern) => pattern.test(gpu));
  return rank === -1 ? GPU_POWER_ORDER.length : rank;
}

function initGpuPicker(root) {
  const select = root.querySelector("[data-gpu-select]");
  const panels = Array.from(root.querySelectorAll("[data-gpu-panel]"));
  if (!(select instanceof HTMLSelectElement) || panels.length === 0) {
    return;
  }

  const available = panels
    .map((panel) => panel.getAttribute("data-gpu"))
    .filter((gpu, index, values) => gpu && values.indexOf(gpu) === index)
    .sort((a, b) => getGpuPowerRank(a) - getGpuPowerRank(b) || a.localeCompare(b));

  select.replaceChildren(
    ...available.map((gpu) => {
      const option = document.createElement("option");
      option.value = gpu;
      option.textContent = gpu;
      return option;
    }),
  );

  const stored = localStorage.getItem(STORAGE_KEY);
  const initial = stored && available.includes(stored) ? stored : available[0];

  function showGpu(gpu) {
    panels.forEach((panel) => {
      panel.toggleAttribute("data-gpu-hidden", panel.getAttribute("data-gpu") !== gpu);
    });
    select.value = gpu;
    localStorage.setItem(STORAGE_KEY, gpu);
  }

  select.addEventListener("change", () => showGpu(select.value));
  showGpu(initial);
}

document.querySelectorAll("[data-gpu-picker]").forEach(initGpuPicker);
