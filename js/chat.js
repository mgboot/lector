import { getWord } from "./store.js";
import { renderMarkdown } from "./markdown.js";

const CHAT_API = "http://localhost:8000/api/chat";
const HEALTH_API = "http://localhost:8000/api/health";

let sessionId = null;
let chatReady = false;

// These are set by the reader module when a word is clicked
let selectedWord = null;
let selectedWordData = null;
let textId = null;
let textTitle = null;

export function setSelectedWord(word, data) {
  selectedWord = word;
  selectedWordData = data;
}

export function setTextId(id) {
  textId = id;
}

export function setTextTitle(title) {
  textTitle = title;
}

export function resetChat() {
  sessionId = null;
  selectedWord = null;
  selectedWordData = null;
  textId = null;
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
      if (data.search_available === false) {
        setStatus("Tutor connected, but textbook search is unavailable (index not found).");
      }
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
  if (textId) context.textId = textId;
  if (textTitle) context.textTitle = textTitle;
  if (selectedWord) {
    context.selectedWord = selectedWord;
    const data = getWord(selectedWord);
    if (data) context.wordData = data;
  }

  const body = { message, context };
  if (sessionId) body.session_id = sessionId;

  // Show thinking indicator
  const bubble = addMessage("assistant", "…");
  bubble.classList.add("thinking");

  let res;
  try {
    res = await fetch(CHAT_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    bubble.textContent = "Unable to reach the tutor server. Is it running?";
    bubble.classList.remove("thinking");
    return;
  }

  if (!res.ok) {
    bubble.textContent = `Server error: ${res.status}`;
    bubble.classList.remove("thinking");
    return;
  }

  const data = await res.json();

  if (data.session_id) sessionId = data.session_id;

  // Render tool-call indicators before the reply
  if (data.tool_calls && data.tool_calls.length > 0) {
    const container = document.getElementById("chat-messages");
    // Insert tool indicators just before the thinking bubble
    for (const tc of data.tool_calls) {
      const indicator = document.createElement("div");
      indicator.className = "tool-indicator";
      indicator.innerHTML = `<span class="tool-icon">${tc.icon || "🔧"}</span><span>${tc.label || tc.tool_call}</span>`;
      container.insertBefore(indicator, bubble);
    }
  }

  // Render the reply
  bubble.classList.remove("thinking");
  if (data.reply) {
    bubble.innerHTML = renderMarkdown(data.reply);
  } else {
    bubble.textContent = "The tutor didn't respond. Check that the server is running and configured correctly.";
  }

  const container = document.getElementById("chat-messages");
  container.scrollTop = container.scrollHeight;
}
