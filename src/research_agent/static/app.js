// Research agent UI: consumes POST /research/stream (SSE) and renders progress + report.
// No business logic here: the API is the source of truth. All model/web text is escaped
// before any markdown formatting is applied, so nothing from the stream becomes raw HTML.

const $ = (id) => document.getElementById(id);
const form = $("ask"), question = $("question"), runBtn = $("run"), newThreadBtn = $("new-thread");
const trace = $("trace"), report = $("report"), usageEl = $("usage"), errorEl = $("error");
const threadNote = $("thread-note");

let threadId = null;
let subQuestions = [];        // index i -> text of sq{i+1}
let steps = {};               // node step elements
let lanes = {};               // "sq1:1" -> lane element

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// --- Markdown (small, safe subset: headings, lists, bold, links, [n] citations) ---
function inline(text, validIds) {
  let s = escapeHtml(text);
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_, label, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
  s = s.replace(/\[(\d{1,3})\]/g, (m, n) =>
    validIds.has(Number(n)) ? `<a class="cite" href="#src-${n}" aria-label="Source ${n}">[${n}]</a>` : m);
  return s;
}

function renderMarkdown(md, sources) {
  const validIds = new Set(sources.map((s) => s.id));
  const body = md.split(/\n## Sources\n/)[0];
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of body.split("\n")) {
    const line = raw.trimEnd();
    let m;
    if ((m = line.match(/^(#{2,3})\s+(.*)$/))) {
      closeList();
      const level = m[1].length;
      out.push(`<h${level}>${inline(m[2], validIds)}</h${level}>`);
    } else if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1], validIds)}</li>`);
    } else if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1], validIds)}</li>`);
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList();
      out.push(`<p>${inline(line, validIds)}</p>`);
    }
  }
  closeList();
  if (sources.length) {
    out.push("<h2>Sources</h2><ol class=\"sources\">");
    for (const s of sources) {
      const title = escapeHtml(s.title || s.url);
      const link = /^https?:\/\//.test(s.url)  // never render javascript: or data: links
        ? `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener noreferrer">${title}</a>`
        : title;
      out.push(`<li id="src-${s.id}">[${s.id}] ${link}</li>`);
    }
    out.push("</ol>");
  }
  return out.join("\n");
}

// --- Trace ---
const NODE_TITLES = {
  planner: "Planning sub-questions",
  research: "Researching in parallel",
  analysis: "Checking coverage",
  answer: "Writing the report",
};

function step(node, iteration) {
  const key = node === "research" || node === "analysis" ? `${node}:${iteration}` : node;
  if (!steps[key]) {
    const li = el("li", "step running");
    let title = NODE_TITLES[node] || node;
    if (iteration > 1 && node === "research") title = "Researching gaps again";
    li.append(el("div", "step-title", title), el("div", "step-meta"));
    if (node === "research") li.append(el("ul", "lanes"));
    trace.append(li);
    steps[key] = li;
  }
  return steps[key];
}

function lane(sqId, iteration) {
  const key = `${sqId}:${iteration}`;
  if (!lanes[key]) {
    const stepEl = step("research", iteration);
    const idx = Number(sqId.replace("sq", "")) - 1;
    const li = el("li", `lane running${iteration > 1 ? " retry" : ""}`);
    li.append(el("p", "lane-q", subQuestions[idx] || sqId));
    if (iteration > 1) li.append(el("div", "lane-tag", "Filling gaps found in the first round"));
    li.append(el("ul", "lane-log"));
    stepEl.querySelector(".lanes").append(li);
    lanes[key] = li;
  }
  return lanes[key];
}

const lastLane = {};  // sub_question_id -> latest iteration seen
function logTo(sqId, text) {
  const li = lane(sqId, lastLane[sqId] || 1);
  li.querySelector(".lane-log").append(el("li", "", text));
}

// Research steps have no finish event of their own (their lanes do); close them when the
// next node starts.
function closeResearchSteps() {
  for (const [key, li] of Object.entries(steps)) {
    if (!key.startsWith("research:") || !li.classList.contains("running")) continue;
    li.classList.replace("running", "done");
    const n = li.querySelectorAll(".lane").length;
    li.querySelector(".step-meta").textContent = `${n} sub-question${n === 1 ? "" : "s"} researched`;
  }
}

function onNode(d) {
  if (d.node === "research") {
    lastLane[d.sub_question_id] = d.iteration;
    const l = lane(d.sub_question_id, d.iteration);
    if (d.status === "finished") {
      l.classList.remove("running");
      if (d.failed) { l.classList.add("failed"); logTo(d.sub_question_id, "No usable sources found"); }
      else logTo(d.sub_question_id, `Done: ${d.sources} sources`);
    }
    return;
  }
  if (d.status === "started") closeResearchSteps();
  const s = step(d.node, d.iteration || 1);
  if (d.status !== "finished") return;
  s.classList.replace("running", "done");
  const meta = s.querySelector(".step-meta");
  if (d.node === "planner" && d.sub_questions) {
    subQuestions = d.sub_questions;
    meta.textContent = `${d.sub_questions.length} sub-questions`;
  } else if (d.node === "analysis") {
    const cov = d.coverage ? Object.values(d.coverage) : [];
    meta.textContent = d.skipped ? "Iteration limit reached"
      : cov.length ? `${cov.filter((c) => c === "sufficient").length} of ${cov.length} covered` : "";
  } else if (d.node === "answer") {
    meta.textContent = `${d.cited_sources ?? 0} sources cited`;
  }
  if (d.latency_ms) meta.textContent += `${meta.textContent ? ", " : ""}${(d.latency_ms / 1000).toFixed(1)} s`;
}

// --- SSE over fetch (EventSource cannot POST) ---
async function* sse(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

function reset() {
  trace.replaceChildren(); report.replaceChildren(); usageEl.textContent = "";
  errorEl.hidden = true; steps = {}; lanes = {}; subQuestions = [];
  for (const k of Object.keys(lastLane)) delete lastLane[k];
}

function showError(message) { errorEl.textContent = message; errorEl.hidden = false; }

async function run(q) {
  reset();
  runBtn.disabled = true; runBtn.textContent = "Researching…";
  let streamed = "";
  try {
    const res = await fetch("/research/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(threadId ? { question: q, thread_id: threadId } : { question: q }),
    });
    if (res.status === 409) return showError("This thread is still working on a question. Wait for it to finish.");
    if (res.status === 422) return showError("Enter a question between 3 and 2000 characters.");
    if (!res.ok) return showError(`The server returned ${res.status}. Check the service logs.`);
    for await (const { event, data } of sse(res)) {
      if (event === "run_started") threadId = data.thread_id;
      else if (event === "node") onNode(data);
      else if (event === "search") logTo(data.sub_question_id, `Searched “${data.query}”${data.cached ? " (cached)" : ""}`);
      else if (event === "fetch") logTo(data.sub_question_id, `Read ${new URL(data.url).hostname}`);
      else if (event === "tool_error") logTo(data.sub_question_id, data.kind === "budget" ? "Search budget used up" : `${data.tool} failed`);
      else if (event === "answer_reset") { streamed = ""; report.textContent = ""; }
      else if (event === "token") {
        streamed += data.text;
        report.className = "prose streaming"; report.textContent = streamed;
      } else if (event === "report") {
        report.className = "prose";
        report.innerHTML = renderMarkdown(data.report.markdown, data.report.sources);
        const u = data.usage;
        const cost = u.estimated_cost_usd == null ? "" : `, about $${u.estimated_cost_usd.toFixed(3)}`;
        usageEl.textContent = `${u.searches} searches, ${u.fetches} pages read, ` +
          `${(u.input_tokens + u.output_tokens).toLocaleString()} tokens${cost}, ${u.latency_s.toFixed(0)} s.`;
      } else if (event === "error") showError(`The run failed: ${data.message}`);
    }
  } catch (e) {
    showError("Lost the connection to the server. Check that it is running and try again.");
  } finally {
    runBtn.disabled = false; runBtn.textContent = threadId ? "Ask a follow-up" : "Research";
    newThreadBtn.hidden = !threadId;
    threadNote.textContent = threadId ? "Follow-ups reuse what this thread already found." : "";
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = question.value.trim() || question.placeholder;
  question.value = q;
  run(q);
});
newThreadBtn.addEventListener("click", () => {
  threadId = null; newThreadBtn.hidden = true; threadNote.textContent = "";
  runBtn.textContent = "Research"; question.value = ""; question.focus();
});
