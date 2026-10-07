import { useState } from "react";
import "./App.css";

function App() {
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState([]);

  const [screenshotSize, setScreenshotSize] = useState({
    width: 0,
    height: 0,
  });

  // ============================================================
  // ANALYZE CURRENT PAGE
  // ============================================================

  const analyzePage = async () => {
    setStatus("scanning");
    setResults([]);

    try {
      // Capture the currently visible browser tab
      const dataUrl = await chrome.tabs.captureVisibleTab(null, {
        format: "png",
      });

      // Read screenshot dimensions
      const image = new Image();
      image.src = dataUrl;

      await new Promise((resolve, reject) => {
        image.onload = resolve;
        image.onerror = reject;
      });

      setScreenshotSize({
        width: image.naturalWidth,
        height: image.naturalHeight,
      });

      // Convert data URL to Blob
      const imageResponse = await fetch(dataUrl);
      const imageBlob = await imageResponse.blob();

      // Create multipart form data
      const formData = new FormData();
      formData.append(
        "image",
        imageBlob,
        "darpan-page.png"
      );

      // Send screenshot to FastAPI
      const response = await fetch(
        "http://127.0.0.1:8000/api/analyze",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Backend returned HTTP ${response.status}`
        );
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(
          data.message || "Analysis failed."
        );
      }

      // Store real backend results
      setResults(data.patterns || []);
      setStatus("complete");

    } catch (error) {
      console.error(
        "DARPAN analysis error:",
        error
      );

      setResults([]);
      setStatus("error");
    }
  };

  // ============================================================
  // HIGHLIGHT FINDING ON WEBPAGE
  // ============================================================

  const highlightElement = async (result) => {
    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });

      if (!tab?.id) {
        console.error("DARPAN: Active tab not found.");
        return;
      }

      const url = tab.url || "";
      if (
        url.startsWith("chrome://") ||
        url.startsWith("chrome-extension://") ||
        url.startsWith("edge://") ||
        url.startsWith("about:") ||
        url.startsWith("view-source:") ||
        url.includes("chromewebstore.google.com")
      ) {
        console.warn("DARPAN: Highlighting is unavailable on internal browser pages.");
        return;
      }

      // Safely pull bounding box (nested or flat)
      const boundingBox =
        result?.bounding_box ||
        result?.visual?.bounding_box ||
        null;

      // Safely pull text evidence
      const supportingText =
        result?.supporting_text ||
        result?.text?.supporting_text ||
        result?.text ||
        result?.evidence ||
        "";

      const payload = {
        action: "highlight",
        result: {
          ...result,
          supporting_text: supportingText,
          bounding_box: boundingBox,
        },
        imageWidth: Number(screenshotSize.width) || 0,
        imageHeight: Number(screenshotSize.height) || 0,
      };

      const sendToTab = async () => {
        return await chrome.tabs.sendMessage(tab.id, payload);
      };

      try {
        await sendToTab();
        console.log("DARPAN: Highlight message sent successfully.");
      } catch (firstError) {
        console.warn(
          "DARPAN: Connection failed. Injecting content script into active tab...",
          firstError
        );

        // Inject content.js dynamically into all frames
        await chrome.scripting.executeScript({
          target: { tabId: tab.id, allFrames: true },
          files: ["content.js"],
        });

        // Small delay to allow listener registration
        await new Promise((resolve) => setTimeout(resolve, 80));

        // Retry sending message
        await sendToTab();
        console.log("DARPAN: Highlight message dispatched after dynamic injection.");
      }
    } catch (error) {
      console.error("DARPAN highlighting failed:", error);
    }
  };

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="header">
        <div className="header-brand">
          <div className="brand-mark">◈</div>
          <div>
            <h1>DARPAN</h1>
            <p>Ethical UI Audit</p>
          </div>
        </div>

        <div className="header-status">
          <span className="status-dot" />
          READY
        </div>
      </header>

      <main>

        {/* ====================================================
            IDLE STATE
        ==================================================== */}

        {status === "idle" && (
          <section className="welcome">
            <div className="hero-icon">◈</div>

            <h2>Analyze this page</h2>

            <p>
              Detect potentially deceptive UI patterns
              and receive ethical redesign recommendations.
            </p>

            <button
              className="primary-button"
              onClick={analyzePage}
            >
              Analyze Page
            </button>

            <div className="feature-strip">
              <div className="feature-item">
                <strong>Visual analysis</strong>
                <span>Interface patterns</span>
              </div>

              <div className="feature-item">
                <strong>Text analysis</strong>
                <span>Language cues</span>
              </div>

              <div className="feature-item">
                <strong>Evidence</strong>
                <span>Supporting signals</span>
              </div>

              <div className="feature-item">
                <strong>Ethical guidance</strong>
                <span>Redesign suggestions</span>
              </div>
            </div>
          </section>
        )}

        {/* ====================================================
            ANIMATED LENS SCANNING STATE
        ==================================================== */}

        {status === "scanning" && (
          <section className="lens-scanner-container">
            <div className="optical-viewfinder">
              {/* Outer Rotating Calibration Rings */}
              <div className="lens-ring ring-outer"></div>
              <div className="lens-ring ring-inner"></div>

              {/* Crosshair Grids */}
              <div className="crosshair-h"></div>
              <div className="crosshair-v"></div>

              {/* Laser Scanning Beam */}
              <div className="laser-beam"></div>

              {/* Central Mirror / Lens Core */}
              <div className="lens-core">
                <div className="lens-reflection"></div>
                <div className="core-dot"></div>
              </div>
            </div>

            <h2 className="scanner-headline">DARPAN Lens Active</h2>
            <p className="scanner-subtext">
              Sneaking through visual nodes, typography & pricing cues...
            </p>

            <div className="optical-telemetry">
              <div className="telemetry-bar">
                <div className="telemetry-progress"></div>
              </div>
              <div className="telemetry-labels">
                <span>YOLOv8 + OCR + DISTILBERT</span>
                <span className="status-ping">● AUDITING</span>
              </div>
            </div>
          </section>
        )}

        {/* ====================================================
            ERROR STATE
        ==================================================== */}

        {status === "error" && (
          <section className="scanning">
            <h2>Analysis failed</h2>

            <div className="error-message">
              DARPAN could not analyze this page.
              Please make sure the backend is running
              and try again.
            </div>

            <button
              className="primary-button"
              onClick={analyzePage}
              style={{ marginTop: "16px" }}
            >
              Try Again
            </button>
          </section>
        )}

        {/* ====================================================
            COMPLETE / RESULTS STATE
        ==================================================== */}

        {status === "complete" && (
          <section className="results">
            <div className="summary-row">
              <div className="summary">
                <h2>Potential issues</h2>
                <p>
                  {results.length > 0
                    ? `DARPAN found ${results.length} possible design pattern${
                        results.length > 1 ? "s" : ""
                      } on this page.`
                    : "No strong evidence of deceptive interface design was found."}
                </p>
              </div>

              <div className="result-count">
                {results.length}
              </div>
            </div>

            {/* Empty State */}
            {results.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">✓</div>
                <h3>No strong patterns detected</h3>
                <p>
                  DARPAN did not find sufficient
                  evidence of a deceptive design
                  pattern on this page.
                </p>
              </div>
            )}

            {/* Result Cards */}
            {results.map((result, index) => (
              <article
                key={
                  result.id ||
                  `${result.label || "pattern"}-${index}`
                }
                className="result-card"
                onClick={() => highlightElement(result)}
              >
                <div className="result-top">
                  <span className="warning">⚠</span>
                  <div>
                    <h3>
                      {result.label ||
                        result.pattern ||
                        "Detected pattern"}
                    </h3>
                    <span className="confidence">
                      Confidence:{" "}
                      {(
                        (result.confidence ?? 0) * 100
                      ).toFixed(0)}
                      %
                    </span>
                  </div>
                </div>

                <div className="evidence">
                  “
                  {result.supporting_text ||
                    result.text ||
                    "No supporting text available"}
                  ”
                </div>

                <div className="section">
                  <h4>Why it matters</h4>
                  <p>
                    {result.why_it_matters ||
                      "This interface element may influence users in a way that is not sufficiently transparent."}
                  </p>
                </div>

                <div className="section recommendation">
                  <h4>Ethical recommendation</h4>
                  <p>
                    {result.ethical_recommendation ||
                      result.recommendation ||
                      "Use clear, transparent, and user-respecting interface design."}
                  </p>
                </div>

                <button
                  className="secondary-button"
                  onClick={(event) => {
                    event.stopPropagation();
                    highlightElement(result);
                  }}
                >
                  Show on page
                </button>
              </article>
            ))}

            <button
              className="primary-button"
              style={{ marginTop: "8px" }}
              onClick={analyzePage}
            >
              Analyze Again
            </button>
          </section>
        )}

      </main>

      {/* ======================================================
          FOOTER
      ====================================================== */}

      <footer>
        <span>DARPAN</span>
        <span>Ethical Design Assistant</span>
      </footer>

    </div>
  );
}

export default App;