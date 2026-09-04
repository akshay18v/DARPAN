let currentOverlay = null;
let currentTargetElement = null;
let cleanupTimer = null;

function removeHighlight() {
  if (cleanupTimer) {
    clearTimeout(cleanupTimer);
    cleanupTimer = null;
  }

  if (currentOverlay) {
    currentOverlay.remove();
    currentOverlay = null;
  }

  if (currentTargetElement) {
    currentTargetElement.style.outline = "";
    currentTargetElement.style.outlineOffset = "";
    currentTargetElement.style.backgroundColor = "";
    currentTargetElement = null;
  }

  document.querySelectorAll("[data-darpan-overlay='true']").forEach((el) => el.remove());
}

function normalize(str) {
  return (str || "")
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    // Retain currency signs, math operators, numbers, and letters so fee strings remain intact
    .replace(/[^a-z0-9+₹$€£%]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// 1. Text Search Strategy (Handles Standard Cues, Hidden Fees & Fragmented Nodes)
function highlightByText(rawTarget) {
  const target = normalize(rawTarget);
  if (!target || target.length < 2) return false;

  const candidateElements = Array.from(
    document.querySelectorAll(
      "p, span, h1, h2, h3, h4, h5, h6, b, strong, em, button, a, div, td, label, small"
    )
  );

  let bestMatch = null;
  let minLength = Infinity;

  // Phase A: Direct Substring Match
  for (const el of candidateElements) {
    if (
      el.getAttribute("data-darpan-overlay") ||
      el.offsetParent === null ||
      ["SCRIPT", "STYLE", "NOSCRIPT"].includes(el.tagName)
    ) {
      continue;
    }

    const elementText = normalize(el.innerText || el.textContent);
    if (!elementText) continue;

    if (elementText.includes(target)) {
      // Prioritize the closest leaf container containing the text
      if (elementText.length < minLength) {
        minLength = elementText.length;
        bestMatch = el;
      }
    }
  }

  // Phase B: Hidden Fees & Price/Numeric Context Match
  if (!bestMatch) {
    const tokens = target.split(" ").filter((t) => t.length >= 2);
    const numericToken = tokens.find((t) => /\d+/.test(t));
    const feeKeywords = ["tax", "taxes", "fee", "fees", "charge", "charges", "night", "total", "extra", "plus"];

    if (tokens.length >= 2) {
      for (const el of candidateElements) {
        if (
          el.getAttribute("data-darpan-overlay") ||
          el.offsetParent === null ||
          ["SCRIPT", "STYLE", "NOSCRIPT"].includes(el.tagName)
        ) {
          continue;
        }

        const text = normalize(el.innerText || el.textContent);
        if (!text) continue;

        // Case 1: Numeric fee amount + fee keyword in the same block (e.g., "450" and "taxes")
        if (numericToken && text.includes(numericToken)) {
          const hasFeeContext = feeKeywords.some((kw) => text.includes(kw));
          if (hasFeeContext && text.length < minLength) {
            minLength = text.length;
            bestMatch = el;
            continue;
          }
        }

        // Case 2: Multi-token match across split spans/tags
        const matchedTokens = tokens.filter((tok) => text.includes(tok));
        if (matchedTokens.length / tokens.length >= 0.6) {
          if (text.length < minLength) {
            minLength = text.length;
            bestMatch = el;
          }
        }
      }
    }
  }

  if (bestMatch) {
    renderDomElementHighlight(bestMatch);
    return true;
  }

  return false;
}

function renderDomElementHighlight(element) {
  removeHighlight();
  currentTargetElement = element;

  element.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });

  element.style.outline = "3px solid #ff4d4f";
  element.style.outlineOffset = "3px";
  element.style.backgroundColor = "rgba(255, 77, 79, 0.12)";

  // Badge label
  const rect = element.getBoundingClientRect();
  const label = document.createElement("div");
  label.setAttribute("data-darpan-overlay", "true");
  label.textContent = "DARPAN • Detected Pattern";

  Object.assign(label.style, {
    position: "absolute",
    left: `${Math.max(8, window.scrollX + rect.left)}px`,
    top: `${Math.max(8, window.scrollY + rect.top - 28)}px`,
    zIndex: "2147483647",
    background: "#0f172a",
    color: "#ffffff",
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "11px",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    fontWeight: "600",
    boxShadow: "0 2px 8px rgba(0,0,0,0.24)",
    pointerEvents: "none",
  });

  document.body.appendChild(label);
  currentOverlay = label;

  cleanupTimer = setTimeout(removeHighlight, 10000);
}

// 2. Bounding Box Fallback Strategy
function highlightByBoundingBox(boundingBox, imageWidth, imageHeight) {
  if (!boundingBox) return false;

  const { x1, y1, x2, y2 } = boundingBox;
  if (![x1, y1, x2, y2].every(Number.isFinite)) return false;

  removeHighlight();

  // Screen-to-CSS pixel scaling factor
  const dpr = window.devicePixelRatio || 1;
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;

  // Use image dimension ratio or fallback to dpr
  const scaleX = imageWidth > 0 ? viewportWidth / (imageWidth / dpr) : 1;
  const scaleY = imageHeight > 0 ? viewportHeight / (imageHeight / dpr) : 1;

  const left = (x1 / dpr) * scaleX;
  const top = (y1 / dpr) * scaleY;
  const width = ((x2 - x1) / dpr) * scaleX;
  const height = ((y2 - y1) / dpr) * scaleY;

  const overlay = document.createElement("div");
  overlay.setAttribute("data-darpan-overlay", "true");

  Object.assign(overlay.style, {
    position: "absolute",
    left: `${window.scrollX + left}px`,
    top: `${window.scrollY + top}px`,
    width: `${Math.max(16, width)}px`,
    height: `${Math.max(16, height)}px`,
    border: "3px dashed #ff4d4f",
    background: "rgba(255, 77, 79, 0.15)",
    boxShadow: "0 0 0 4px rgba(255, 77, 79, 0.1), 0 4px 16px rgba(0,0,0,0.2)",
    borderRadius: "6px",
    zIndex: "2147483647",
    pointerEvents: "none",
    boxSizing: "border-box",
  });

  // Small badge label
  const label = document.createElement("div");
  label.textContent = "DARPAN • Visual Detection";
  Object.assign(label.style, {
    position: "absolute",
    top: "-24px",
    left: "0",
    background: "#ff4d4f",
    color: "#ffffff",
    padding: "2px 6px",
    fontSize: "10px",
    fontWeight: "700",
    borderRadius: "3px",
    whiteSpace: "nowrap",
  });

  overlay.appendChild(label);
  document.body.appendChild(overlay);
  currentOverlay = overlay;

  window.scrollTo({
    top: Math.max(0, window.scrollY + top - window.innerHeight / 3),
    behavior: "smooth",
  });

  cleanupTimer = setTimeout(removeHighlight, 10000);
  return true;
}

// 3. Message Dispatcher
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.action !== "highlight") {
    if (message?.action === "clearHighlight") removeHighlight();
    return;
  }

  const result = message.result || {};
  const text = result.supporting_text || result.text?.supporting_text || result.text || "";
  const box = result.bounding_box || result.visual?.bounding_box || null;

  // Step 1: Text-based search
  const textFound = highlightByText(text);
  if (textFound) {
    sendResponse({ status: "success", strategy: "text" });
    return true;
  }

  // Step 2: Bounding Box Fallback
  const boxFound = highlightByBoundingBox(
    box,
    Number(message.imageWidth) || 0,
    Number(message.imageHeight) || 0
  );

  if (boxFound) {
    sendResponse({ status: "success", strategy: "bbox" });
    return true;
  }

  console.warn("DARPAN: Element could not be matched via DOM text or bounding box.");
  sendResponse({ status: "not_found" });
  return true;
});