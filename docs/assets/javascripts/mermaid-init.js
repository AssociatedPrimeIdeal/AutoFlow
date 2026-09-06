document.addEventListener("DOMContentLoaded", () => {
  const diagrams = document.querySelectorAll(".language-mermaid");
  if (!diagrams.length) return;

  import("https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs")
    .then(({ default: mermaid }) => {
      mermaid.initialize({ startOnLoad: false, securityLevel: "loose", theme: "neutral" });
      return mermaid.run({ nodes: diagrams });
    })
    .catch((error) => console.warn("Mermaid diagrams could not be loaded", error));
});
