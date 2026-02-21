import { getWord } from "./store.js";
import { renderMarkdown } from "./markdown.js";

const CHAT_API = "http://localhost:8000/api/chat";
const HEALTH_API = "http://localhost:8000/api/health";

let sessionId = null;
let chatReady = false;

// These are set by the reader module when a word is clicked
let selectedWord = null;
let selectedWordData = null;
let passageText = null;
let textTitle = null;

export function setSelectedWord(word, data) {
  selectedWord = word;
  selectedWordData = data;
}

export function setPassageText(text) {
  passageText = text;
}

export function setTextTitle(title) {
  textTitle = title;
}

export function resetChat() {
  sessionId = null;
  selectedWord = null;
  selectedWordData = null;
  passageText = null;
  textTitle = null;

  const container = document.getElementById("chat-messages");
  container.innerHTML = '<div class="chat-placeholder">Ask a question about the text or a selected word…</div>';
}

export function initChat() {
  const form = document.getElementById("chat-form");
  form.addEventListener("submit", onSubmit);
  checkHealth();
}

async function checkHealth() {
  setStatus("Connecting to tutor…");
  try {
    const res = await fetch(HEALTH_API);
    const data = await res.json();
    if (data.status === "ok") {
      chatReady = true;
      clearStatus();
    } else {
      setStatus(`Tutor unavailable: ${data.detail || "unknown error"}`);
    }
  } catch {
    setStatus("Could not reach the tutor server. Is it running?");
  }
}

function setStatus(text) {
  let el = document.getElementById("chat-status");
  if (!el) {
    el = document.createElement("div");
    el.id = "chat-status";
    el.className = "chat-status";
    const container = document.getElementById("chat-messages");
    container.appendChild(el);
  }
  el.textContent = text;
  el.hidden = false;
}

function clearStatus() {
  const el = document.getElementById("chat-status");
  if (el) el.hidden = true;
}

async function onSubmit(e) {
  e.preventDefault();
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  addMessage("user", message);

  const sendBtn = document.getElementById("chat-send");
  sendBtn.disabled = true;

  try {
    await sendMessage(message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function addMessage(role, text) {
  const container = document.getElementById("chat-messages");

  // Remove placeholder if present
  const placeholder = container.querySelector(".chat-placeholder");
  if (placeholder) placeholder.remove();

  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  container.appendChild(bubble);
  container.scrollTop = container.scrollHeight;
  return bubble;
}

async function sendMessage(message) {
  const context = {};
  if (passageText) context.text = passageText;
  if (textTitle) context.textTitle = textTitle;
  if (selectedWord) {
    context.selectedWord = selectedWord;
    const data = getWord(selectedWord);
    if (data) context.wordData = data;
  }

  const body = { message, context };
  if (sessionId) body.session_id = sessionId;

  let res;
  try {
    res = await fetch(CHAT_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    addMessage("assistant", "Unable to reach the tutor server. Is it running?");
    return;
  }

  if (!res.ok) {
    addMessage("assistant", `Server error: ${res.status}`);
    return;
  }

  // Stream the response
  const bubble = addMessage("assistant", "Thinking…");
  bubble.classList.add("thinking");
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let firstChunk = true;
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    let text = decoder.decode(value, { stream: true });

    // First chunk contains the session_id JSON line
    if (firstChunk) {
      const nlIdx = text.indexOf("\n");
      if (nlIdx !== -1) {
        try {
          const meta = JSON.parse(text.slice(0, nlIdx));
          if (meta.session_id) sessionId = meta.session_id;
        } catch { /* ignore parse errors */ }
        text = text.slice(nlIdx + 1);
      }
      firstChunk = false;
    }

    fullText += text;
    bubble.innerHTML = renderMarkdown(fullText);
    bubble.classList.remove("thinking");
    const container = document.getElementById("chat-messages");
    container.scrollTop = container.scrollHeight;
  }

  if (!fullText) {
    bubble.textContent = "The tutor didn't respond. Check that the server is running and configured correctly.";
  }
}
