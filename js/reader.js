import { getWord, setWord } from "./store.js";
import { fetchWord } from "./api.js";
import { setSelectedWord } from "./chat.js";

const POS_LABELS = {
  N: "Noun", V: "Verb", ADJ: "Adjective", ADV: "Adverb",
  PRON: "Pronoun", PREP: "Preposition", CONJ: "Conjunction",
  INTERJ: "Interjection", VPAR: "Participle", SUPINE: "Supine",
  NUM: "Numeral", PACKON: "Packon", TACKON: "Tackon",
  PREFIX: "Prefix", SUFFIX: "Suffix", X: "Unknown",
};

const SPACY_POS_LABELS = {
  NOUN: "Noun", VERB: "Verb", ADJ: "Adjective", ADV: "Adverb",
  PROPN: "Proper Noun", PRON: "Pronoun", DET: "Determiner",
  ADP: "Preposition", AUX: "Auxiliary Verb", CCONJ: "Coord. Conjunction",
  SCONJ: "Subord. Conjunction", PART: "Particle", INTJ: "Interjection",
  NUM: "Numeral", PUNCT: "Punctuation", SYM: "Symbol", X: "Other",
};

const MORPH_LABELS = {
  Case: { Nom: "Nominative", Voc: "Vocative", Gen: "Genitive", Dat: "Dative", Acc: "Accusative", Abl: "Ablative", Loc: "Locative" },
  Number: { Sing: "Singular", Plur: "Plural" },
  Gender: { Masc: "Masculine", Fem: "Feminine", Neut: "Neuter" },
  Tense: { Pres: "Present", Past: "Past", Fut: "Future", Pqp: "Pluperfect", Imp: "Imperfect" },
  Mood: { Ind: "Indicative", Sub: "Subjunctive", Imp: "Imperative", Inf: "Infinitive", Part: "Participle", Ger: "Gerund", Gdv: "Gerundive", Sup: "Supine" },
  Voice: { Act: "Active", Pass: "Passive" },
  Person: { 1: "1st Person", 2: "2nd Person", 3: "3rd Person" },
  Degree: { Pos: "Positive", Cmp: "Comparative", Sup: "Superlative" },
};

let activeSpan = null;
let _analysisTokens = null;

/**
 * Store LatinCy analysis tokens for later lookup.
 */
export function setAnalysisData(tokens) {
  _analysisTokens = tokens;
}

/**
 * Clear LatinCy analysis tokens (used when switching texts).
 */
export function clearAnalysisData() {
  _analysisTokens = null;
}

/**
 * Find LatinCy token(s) overlapping the given character range.
 */
function lookupAnalysis(start, end) {
  if (!_analysisTokens) return null;
  const matches = _analysisTokens.filter(
    (t) => t.start < end && t.end > start
  );
  return matches.length > 0 ? matches : null;
}

/**
 * Parse a spaCy morph string like "Case=Acc|Gender=Neut|Number=Plur"
 * into an array of human-readable labels.
 */
function parseMorph(morphStr) {
  if (!morphStr) return [];
  const parts = [];
  for (const pair of morphStr.split("|")) {
    const [key, val] = pair.split("=");
    if (!key || !val) continue;
    const labelMap = MORPH_LABELS[key];
    const label = labelMap ? (labelMap[val] || val) : val;
    parts.push(label);
  }
  return parts;
}

/**
 * Tokenize raw text into an array of { type, value } tokens.
 * type is "word" or "other" (whitespace, punctuation, section numbers).
 */
export function tokenize(text) {
  const tokens = [];
  // Match words, section numbers, or non-word/non-digit chunks
  const regex = /([A-Za-zÀ-ÿ]+)|(\d+)|([^A-Za-zÀ-ÿ\d]+)/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match[1]) {
      tokens.push({ type: "word", value: match[1] });
    } else if (match[2]) {
      tokens.push({ type: "section-number", value: match[2] });
    } else {
      tokens.push({ type: "other", value: match[3] });
    }
  }
  return tokens;
}

/**
 * Render tokenized text into the given container element.
 */
export function renderTokens(container, tokens) {
  container.innerHTML = "";
  let charOffset = 0;
  for (const token of tokens) {
    if (token.type === "word") {
      const span = document.createElement("span");
      span.className = "word";
      span.textContent = token.value;
      span.dataset.word = token.value.toLowerCase();
      span.dataset.start = charOffset;
      span.dataset.end = charOffset + token.value.length;
      span.addEventListener("click", () => onWordClick(span));
      container.appendChild(span);
    } else if (token.type === "section-number") {
      const sup = document.createElement("sup");
      sup.className = "section-number";
      sup.textContent = token.value;
      container.appendChild(sup);
    } else {
      // Split on newlines to insert paragraph breaks
      const parts = token.value.split("\n");
      for (let i = 0; i < parts.length; i++) {
        if (i > 0) {
          const spacer = document.createElement("div");
          spacer.className = "paragraph-break";
          container.appendChild(spacer);
        }
        if (parts[i]) {
          container.appendChild(document.createTextNode(parts[i]));
        }
      }
    }
    charOffset += token.value.length;
  }
}

/**
 * Extract unique normalized words from tokens.
 */
export function uniqueWords(tokens) {
  const set = new Set();
  for (const t of tokens) {
    if (t.type === "word") set.add(t.value.toLowerCase());
  }
  return [...set];
}

function onWordClick(span) {
  if (activeSpan) activeSpan.classList.remove("active");
  activeSpan = span;
  span.classList.add("active");

  const word = span.dataset.word;
  const start = parseInt(span.dataset.start, 10);
  const end = parseInt(span.dataset.end, 10);
  const analysis = lookupAnalysis(start, end);
  const data = getWord(word);
  if (data) {
    setSelectedWord(word, data);
    renderSidebar(span.textContent, data, analysis);
  } else {
    fetchAndRenderWord(span, analysis);
  }
}

async function fetchAndRenderWord(span, analysis) {
  const word = span.dataset.word;
  renderSidebarLoading(span.textContent);
  try {
    const data = await fetchWord(word);
    if (data) setWord(word, data);
    // Only render if this span is still the active one
    if (activeSpan === span) {
      setSelectedWord(word, data);
      renderSidebar(span.textContent, data, analysis);
    }
  } catch (err) {
    console.warn(`[Lector] On-demand fetch failed for "${word}":`, err);
    if (activeSpan === span) {
      renderSidebar(span.textContent, null, analysis);
    }
  }
}

function renderSidebarLoading(displayForm) {
  const placeholder = document.querySelector(".sidebar-placeholder");
  const content = document.getElementById("sidebar-content");
  placeholder.hidden = true;
  content.hidden = false;
  content.innerHTML = `<h2>${displayForm}</h2><p class="not-found">Loading…</p>`;
}

function renderSidebar(displayForm, data, analysis) {
  const placeholder = document.querySelector(".sidebar-placeholder");
  const content = document.getElementById("sidebar-content");

  placeholder.hidden = true;
  content.hidden = false;

  if (!data || data.length === 0) {
    // No Whitaker's data — still show LatinCy analysis if available
    let html = `<h2>${displayForm}</h2>`;
    if (analysis) {
      html += renderContextAnalysis(analysis);
    }
    html += `<p class="not-found">No dictionary data available for this word.</p>`;
    content.innerHTML = html;
    return;
  }

  let html = "";
  for (const entry of data) {
    const rootLine = entry.rootLines?.[0];
    const root = rootLine?.root ?? "";
    const pos = rootLine?.partOfSpeech ?? "";
    const posLabel = POS_LABELS[pos] || pos;
    const version = rootLine?.version ?? "";

    html += `<h2>${displayForm} <span class="pos-tag">${posLabel}</span></h2>`;
    if (root) html += `<p class="root">${root}</p>`;
    if (version) {
      html += `<p class="morphology"><span>${version}</span></p>`;
    }

    // Context-aware analysis from LatinCy (replaces old Form Analysis)
    if (analysis) {
      html += renderContextAnalysis(analysis);
    }

    // Meanings
    if (entry.meanings?.length) {
      html += `<h3>Meanings</h3><ul class="meanings">`;
      for (const m of entry.meanings) {
        const trimmed = m.trim();
        if (trimmed) html += `<li>${trimmed}</li>`;
      }
      html += `</ul>`;
    }
  }

  content.innerHTML = html;
}

/**
 * Render the LatinCy context-aware analysis section.
 */
function renderContextAnalysis(analysisTokens) {
  let html = `<h3>Context Analysis</h3><div class="morphology">`;
  for (const t of analysisTokens) {
    const posLabel = SPACY_POS_LABELS[t.pos] || t.pos;
    html += `<div><strong>${posLabel}</strong></div>`;
    if (t.lemma && t.lemma !== t.text.toLowerCase()) {
      html += `<div>Lemma: <em>${t.lemma}</em></div>`;
    }
    const morphParts = parseMorph(t.morph);
    if (morphParts.length > 0) {
      html += `<div>${morphParts.join(", ")}</div>`;
    }
  }
  html += `</div>`;
  return html;
}
