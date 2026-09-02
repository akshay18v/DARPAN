let currentHighlight = null;

function removeHighlight() {
  if (currentHighlight) {
    currentHighlight.remove();
    currentHighlight = null;
  }
}

function createHighlight(result) {
  removeHighlight();

  if (!result) return;

  /*
   * Temporary implementation:
   * In mock mode we highlight the first visible text
   * that contains the detected OCR text.
   *
   * Later, the backend can return an element identifier
   * or DOM coordinates for more accurate highlighting.
   */

  const targetText = (result.text || "").toLowerCase();

  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT
  );

  let node = null;

  while ((node = walker.nextNode())) {
    const value = node.nodeValue?.trim().toLowerCase();

    if (
      value &&
      targetText &&
      value.includes(targetText)
    ) {
      break;
    }
  }

  if (!node) {
    console.warn(
      "DARPAN could not locate the detected text on the page."
    );
    return;
  }

  const element = node.parentElement;

  if (!element) return;

  const rect = element.getBoundingClientRect();

  const overlay = document.createElement("div");

  overlay.setAttribute(
    "data-darpan-highlight",
    "true"
  );

  Object.assign(
    overlay.style,
    {
      position: "fixed",
      left: `${rect.left - 4}px`,
      top: `${rect.top - 4}px`,
      width: `${rect.width + 8}px`,
      height: `${rect.height + 8}px`,
      border: "3px solid #ff4d4f",
      borderRadius: "6px",
      background: "rgba(255, 77, 79, 0.10)",
      boxShadow: "0 0 0 9999px rgba(0, 0, 0, 0.08)",
      zIndex: "2147483647",
      pointerEvents: "none",
      transition: "all 0.2s ease"
    }
  );

  document.body.appendChild(overlay);

  currentHighlight = overlay;

  element.scrollIntoView({
    behavior: "smooth",
    block: "center"
  });
}

chrome.runtime.onMessage.addListener(
  (message) => {

    if (message.action === "highlight") {
      createHighlight(
        message.result
      );
    }

    if (message.action === "clearHighlight") {
      removeHighlight();
    }
  }
);