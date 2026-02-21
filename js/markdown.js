/**
 * Lightweight markdown-to-HTML renderer for LLM chat output.
 * Handles: bold, italic, inline code, unordered/ordered lists, paragraphs.
 */

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

/**
 * Convert a markdown string to HTML.
 */
export function renderMarkdown(src) {
  if (!src) return "";

  const lines = src.split("\n");
  const out = [];
  let inUl = false;
  let inOl = false;

  function closeList() {
    if (inUl) { out.push("</ul>"); inUl = false; }
    if (inOl) { out.push("</ol>"); inOl = false; }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Unordered list item
    const ulMatch = line.match(/^[\s]*[-*]\s+(.*)/);
    if (ulMatch) {
      if (!inUl) { closeList(); out.push("<ul>"); inUl = true; }
      out.push(`<li>${renderInline(ulMatch[1])}</li>`);
      continue;
    }

    // Ordered list item
    const olMatch = line.match(/^[\s]*\d+\.\s+(.*)/);
    if (olMatch) {
      if (!inOl) { closeList(); out.push("<ol>"); inOl = true; }
      out.push(`<li>${renderInline(olMatch[1])}</li>`);
      continue;
    }

    closeList();

    // Blank line → paragraph break
    if (line.trim() === "") {
      out.push("");
      continue;
    }

    // Heading (### only inside chat — keep small)
    const headingMatch = line.match(/^(#{1,4})\s+(.*)/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      out.push(`<strong>${renderInline(headingMatch[2])}</strong>`);
      continue;
    }

    out.push(renderInline(line));
  }

  closeList();

  // Group consecutive non-empty lines into <p> blocks
  const blocks = [];
  let current = [];
  for (const part of out) {
    if (part === "") {
      if (current.length) { blocks.push(current.join("<br>")); current = []; }
    } else {
      current.push(part);
    }
  }
  if (current.length) blocks.push(current.join("<br>"));

  if (blocks.length <= 1) return blocks[0] || "";
  return blocks.map((b) => `<p>${b}</p>`).join("");
}
