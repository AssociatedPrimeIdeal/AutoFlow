document.addEventListener("DOMContentLoaded", () => {
  const diagrams = document.querySelectorAll(".autoflow-mermaid");
  if (!diagrams.length || !window.mermaid) return;
  for (const diagram of diagrams) {
    const code = diagram.querySelector("code");
    if (code) diagram.textContent = code.textContent;
  }
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "neutral",
    fontFamily: "Arial, sans-serif",
  });
  window.mermaid.run({ nodes: diagrams }).catch((error) => {
    console.warn("Mermaid diagram could not be rendered", error);
  });
});
