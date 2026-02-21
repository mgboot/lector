const API_BASE = "https://whitakers-words.com/api/translate/latin";
const PROXY_URL = "https://api.allorigins.win/raw?url=";
const REQUEST_DELAY_MS = 200;

export async function fetchWord(word) {
  const target = `${API_BASE}/${encodeURIComponent(word.toLowerCase())}`;
  const url = `${PROXY_URL}${encodeURIComponent(target)}`;
  const res = await fetch(url);
  if (!res.ok) return null;
  const data = await res.json();
  return Array.isArray(data) && data.length > 0 ? data : null;
}

/**
 * Run LatinCy analysis on the full passage text.
 * Returns an array of token objects with { text, lemma, pos, tag, morph, start, end }.
 */
export async function analyzeText(text) {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) return null;
  const data = await res.json();
  return data.tokens || null;
}

/**
 * Fetch definitions for a list of words with rate-limiting.
 * Calls onProgress(done, total) after each word.
 * Returns a Map<string, apiResponse>.
 */
export async function fetchAllWords(words, onProgress) {
  const results = new Map();
  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    try {
      const data = await fetchWord(word);
      if (data) results.set(word, data);
    } catch (err) {
      console.warn(`[Lector] Failed to fetch "${word}":`, err);
    }
    if (onProgress) onProgress(i + 1, words.length);
    if (i < words.length - 1) {
      await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
    }
  }
  return results;
}
