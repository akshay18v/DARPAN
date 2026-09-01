import { useState } from "react";
import "./App.css";

const MOCK_RESULTS = [
  {
    id: "1",
    class_id: 2,
    pattern: "False Urgency",
    confidence: 0.89,
    text: "Only 2 rooms left",
    why_it_matters:
      "Emphasizes scarcity and may pressure users into making a rushed decision.",
    recommendation:
      "Use accurate availability information without unnecessary urgency-inducing wording.",
  },
  {
    id: "2",
    class_id: 3,
    pattern: "Sneaking",
    confidence: 0.84,
    text: "Protection automatically added to your booking",
    why_it_matters:
      "An optional service appears to be included without requiring a clear affirmative choice.",
    recommendation:
      "Keep optional services unselected by default and require explicit user consent.",
  },
];

function App() {
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState([]);

  const analyzePage = async () => {
    setStatus("scanning");
    setResults([]);

    // Temporary mock result.
    // Later replace this with the FastAPI API call.
    setTimeout(() => {
      setResults(MOCK_RESULTS);
      setStatus("complete");
    }, 1500);
  };

  const highlightElement = async (result) => {
    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });

      if (!tab?.id) return;

      await chrome.tabs.sendMessage(tab.id, {
        action: "highlight",
        result,
      });
    } catch (error) {
      console.error("Could not highlight element:", error);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>DARPAN</h1>
          <p>Ethical UI Audit</p>
        </div>

        <span className="status-dot" />
      </header>

      <main>
        {status === "idle" && (
          <section className="welcome">
            <div className="hero-icon">◈</div>

            <h2>Analyze this page</h2>

            <p>
              Detect potentially deceptive UI patterns and receive
              ethical redesign recommendations.
            </p>

            <button
              className="primary-button"
              onClick={analyzePage}
            >
              Analyze Page
            </button>
          </section>
        )}

        {status === "scanning" && (
          <section className="scanning">
            <div className="loader" />

            <h2>Analyzing page...</h2>

            <p>
              Reviewing the interface for potential dark patterns.
            </p>

            <div className="scan-steps">
              <span>✓ Page captured</span>
              <span>• Analyzing interface</span>
              <span>• Reviewing content</span>
            </div>
          </section>
        )}

        {status === "complete" && (
          <section className="results">
            <div className="summary">
              <h2>
                {results.length > 0
                  ? `${results.length} potential issue${
                      results.length > 1 ? "s" : ""
                    } found`
                  : "No strong dark patterns detected"}
              </h2>

              {results.length === 0 && (
                <p>
                  The page did not provide strong evidence of a
                  deceptive design pattern.
                </p>
              )}
            </div>

            {results.map((result) => (
              <article
                key={result.id}
                className="result-card"
                onClick={() => highlightElement(result)}
              >
                <div className="result-top">
                  <span className="warning">⚠</span>

                  <div>
                    <h3>{result.pattern}</h3>
                    <span className="confidence">
                      Confidence:{" "}
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                <div className="evidence">
                  “{result.text}”
                </div>

                <div className="section">
                  <h4>Why it matters</h4>
                  <p>{result.why_it_matters}</p>
                </div>

                <div className="section recommendation">
                  <h4>Ethical recommendation</h4>
                  <p>{result.recommendation}</p>
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
              className="scan-again"
              onClick={analyzePage}
            >
              Analyze Again
            </button>
          </section>
        )}
      </main>

      <footer>
        <span>DARPAN</span>
        <span>Ethical Design Assistant</span>
      </footer>
    </div>
  );
}

export default App;