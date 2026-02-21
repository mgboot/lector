import { tokenize, renderTokens, uniqueWords, setAnalysisData, clearAnalysisData } from "./reader.js";
import { fetchAllWords, analyzeText } from "./api.js";
import { hasWord, setWord } from "./store.js";
import { initChat, setPassageText, setTextTitle, resetChat } from "./chat.js";

let manifest = [];

async function init() {
  // Load text manifest and populate selector
  const manifestRes = await fetch("texts/manifest.json");
  manifest = await manifestRes.json();

  const selector = document.getElementById("text-selector");
  for (const entry of manifest) {
    const opt = document.createElement("option");
    opt.value = entry.file;
    opt.textContent = entry.title;
    selector.appendChild(opt);
  }

  // Handle dropdown changes (attach before loading so it's responsive immediately)
  selector.addEventListener("change", async () => {
    const entry = manifest.find((e) => e.file === selector.value);
    if (entry) {
      resetChat();
      await loadText(entry);
    }
  });

  // Load the first text before initialising chat so the health-check
  // (which may block the server event loop) doesn't stall static-file serving.
  await loadText(manifest[0]);

  initChat();
}

async function loadText(entry) {
  const res = await fetch(`texts/${entry.file}`);
  const text = await res.text();

  // Update chat context
  setPassageText(text);
  setTextTitle(entry.title);

  // Clear previous analysis data
  clearAnalysisData();

  const tokens = tokenize(text);
  const container = document.getElementById("text-content");
  renderTokens(container, tokens);

  // Run LatinCy analysis on the full text (non-blocking)
  analyzeText(text).then((analysisTokens) => {
    if (analysisTokens) setAnalysisData(analysisTokens);
  });

  // Determine which words still need fetching
  const allWords = uniqueWords(tokens);
  const missing = allWords.filter((w) => !hasWord(w));

  if (missing.length > 0) {
    const indicator = document.getElementById("loading-indicator");
    const progress = document.getElementById("loading-progress");
    indicator.hidden = false;

    const results = await fetchAllWords(missing, (done, total) => {
      progress.value = Math.round((done / total) * 100);
    });

    for (const [word, data] of results) {
      setWord(word, data);
    }

    indicator.hidden = true;
  }
}

init();
