import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const sheet = document.createElement("link");
sheet.rel = "stylesheet";
sheet.href = new URL("./media_editor.css", import.meta.url).href;
document.head.append(sheet);
const BASE = "/yafv/editor";
const mime = "application/x-yafv-media";
const clipMime = "application/x-yafv-clip";
const sessions = new Map();
const sessionFields = ["clips", "undo", "redo", "time", "selected", "zoom", "fps", "width", "height"];
const toolFields = ["color", "brush", "opacity", "scope", "ink-in", "ink-out", "filter"];
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const seconds = x => `${Math.floor(x / 60)}:${(x % 60).toFixed(2).padStart(5, "0")}`;
const uid = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, "0")).join("");

async function request(path, options) {
    const response = await api.fetchApi(BASE + path, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `HTTP ${response.status}`);
    }
    return response;
}
const jsonOptions = data => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = name; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

class Editor {
    constructor(node) {
        this.node = node;
        this.sessionId = uid();
        this.clips = []; this.assets = new Map(); this.metadata = new Map();
        this.undo = []; this.redo = []; this.time = 0; this.selected = null;
        this.zoom = 60; this.fps = 24; this.width = 1280; this.height = 720;
        this.playing = false; this.drawing = false; this.disposed = false;
        this.loaded = null; this.loadVersion = 0; this.frameUrl = null;
        this.video = document.createElement("video");
        this.video.playsInline = true; this.video.preload = "auto";
        this.picture = new Image();
        this.video.addEventListener("seeked", () => this.paint());
        this.video.addEventListener("loadeddata", () => this.paint());
        this.video.addEventListener("error", () => this.message("The browser cannot play this video.", true));
        this.root = document.createElement("div"); this.root.className = "yafv-editor";
        this.root.innerHTML = `
            <header><strong>YAFV · Media Editor</strong><span class="muted">Temporary editing</span><button data-action="fullscreen">Expand</button></header>
            <div class="workspace">
              <aside><b>Queue / results</b><p class="queue muted"></p>
                <div class="tools"><select class="filter"><option value="all">All</option><option value="image">Images</option><option value="video">Videos</option></select><button data-action="refresh">↻</button></div>
                <div class="gallery"></div>
              </aside>
              <section class="monitor">
                <div class="tools"><button data-action="pencil">✎ Pencil</button><input class="color" type="color" value="#ff5252" aria-label="Color">
                  <label>Width <input class="brush" type="number" value="8" min="1" max="200"></label>
                  <label>Opacity <input class="opacity" type="range" min="0.05" max="1" step="0.05" value="1"></label>
                  <select class="scope" aria-label="Drawing duration"><option value="clip">Entire clip</option><option value="frame">Current frame</option><option value="range">Range</option></select>
                  <button data-action="undo">↶</button><button data-action="redo">↷</button><button data-action="clearInk">Clear strokes</button>
                </div>
                <div class="tools range-tools" hidden><label>Draw from <input class="ink-in" type="number" min="0" step="0.01" value="0"></label><label>to <input class="ink-out" type="number" min="0" step="0.01" value="1"></label><span class="muted">seconds within the clip</span></div>
                <div class="stage"><canvas width="1280" height="720"></canvas><div class="hint">Drag an image or video onto the timeline.<br>You can also double-click a result.</div></div>
                <div class="tools transport"><button data-action="previous">−1 frame</button><button data-action="play">▶</button><button data-action="next">+1 frame</button><input class="scrub" type="range" min="0" max="0" step="0.001" value="0" aria-label="Position"><span class="time">0:00.00</span></div>
              </section>
            </div>
            <div class="tools"><b>Timeline</b><button data-action="split">Split</button><button data-action="remove">Delete clip</button><button data-action="duplicate">Duplicate</button>
              <label>In <input class="trim-in" type="number" min="0" step="0.01" value="0"></label><label>Output <input class="trim-out" type="number" min="0" step="0.01" value="0"></label>
              <label>Zoom <input class="zoom" type="range" min="10" max="200" value="60"></label>
            </div>
            <div class="timeline-scroll"><div class="timeline"><div class="ruler"></div><div class="playhead"></div></div></div>
            <footer><label>Output <input class="width" type="number" min="2" max="8192" step="2" value="1280">×<input class="height" type="number" min="2" max="8192" step="2" value="720"></label>
              <label>FPS <input class="fps" type="number" min="1" max="240" value="24"></label>
              <button data-action="extract">Frame → timeline</button><button data-action="png">Download PNG</button><button data-action="export">Export MP4</button><button data-action="cancel" hidden>Cancel</button>
              <progress max="1" value="0" hidden></progress><span class="status" role="status">Drag results onto the timeline to start.</span>
            </footer>`;
        this.$ = selector => this.root.querySelector(selector);
        this.canvas = this.$("canvas"); this.ctx = this.canvas.getContext("2d");
        this.resizeObserver = new ResizeObserver(() => this.fitCanvas());
        this.resizeObserver.observe(this.$(".stage"));
        this.layoutObserver = new ResizeObserver(() => this.scaleInterface());
        this.layoutObserver.observe(this.root);
        this.root.addEventListener("pointerdown", e => e.stopPropagation());
        this.root.addEventListener("wheel", e => e.stopPropagation());
        this.root.addEventListener("keydown", e => {
            e.stopPropagation();
            if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); this.history(e.shiftKey); }
            if (e.code === "Space") { e.preventDefault(); this.togglePlay(); }
        });
        this.root.tabIndex = 0;
        this.root.addEventListener("click", e => {
            const action = e.target.closest("[data-action]")?.dataset.action;
            if (action) this.act(action).catch(error => this.message(error.message, true));
        });
        this.$(".filter").onchange = () => this.gallery();
        this.$(".scope").onchange = () => { this.$(".range-tools").hidden = this.$(".scope").value !== "range"; };
        this.$(".scrub").oninput = e => this.seek(Number(e.target.value));
        this.$(".zoom").oninput = e => { this.zoom = Number(e.target.value); this.timeline(); };
        for (const key of ["width", "height", "fps"]) this.$(`.${key}`).onchange = e => {
            const minimum = key === "fps" ? 1 : 2, maximum = key === "fps" ? 240 : 8192;
            this[key] = clamp(Number(e.target.value) || minimum, minimum, maximum);
            if (key !== "fps") this[key] = Math.ceil(this[key] / 2) * 2;
            e.target.value = this[key]; this.update();
        };
        for (const key of ["in", "out"]) this.$(`.trim-${key}`).onchange = e => {
            const clip = this.selection(); if (!clip) return;
            this.remember(); this.trim(clip, key, Number(e.target.value)); this.update();
        };
        const scroll = this.$(".timeline-scroll");
        scroll.ondragover = e => { e.preventDefault(); scroll.classList.add("drop"); };
        scroll.ondragleave = () => scroll.classList.remove("drop");
        scroll.ondrop = e => {
            e.preventDefault(); e.stopPropagation(); scroll.classList.remove("drop");
            const at = (e.clientX - this.$(".timeline").getBoundingClientRect().left) / this.pixelsPerSecond;
            const target = this.locate(at); const index = target ? this.clips.indexOf(target.clip) : this.clips.length;
            const moving = e.dataTransfer.getData(clipMime), media = e.dataTransfer.getData(mime);
            if (moving) {
                const old = this.clips.findIndex(c => c.uid === moving);
                if (old < 0) return;
                this.remember(); const [clip] = this.clips.splice(old, 1);
                this.clips.splice(index > old ? index - 1 : index, 0, clip); this.update();
            } else if (media) this.add(media, index).catch(error => this.message(error.message, true));
        };
        this.canvas.onpointerdown = e => this.startStroke(e);
        this.canvas.onpointermove = e => this.moveStroke(e);
        this.canvas.onpointerup = e => this.finishStroke(e);
        this.canvas.onpointercancel = e => this.finishStroke(e);
        this.onProgress = ({ detail }) => {
            if (detail.job_id === this.job) { this.$("progress").value = detail.progress; this.message(`Exportando ${Math.round(detail.progress * 100)}%`); }
        };
        this.onExecuted = () => { clearTimeout(this.refreshDelay); this.refreshDelay = setTimeout(() => this.refresh(), 400); };
        api.addEventListener("yafv-editor-progress", this.onProgress);
        api.addEventListener("executed", this.onExecuted);
        this.timer = setInterval(() => this.refresh(), 4000);
        this.refresh();
        this.previousTick = performance.now();
        this.animation = requestAnimationFrame(now => this.tick(now));
        this.widget = node.addDOMWidget("media_editor", "yafv_media_editor", this.root, {
            serialize: false, getMinHeight: () => 560,
        });
        this.widget.serialize = false;
        node.setSize([1120, 860]);
    }
    sessionKey() {
        return JSON.stringify([this.sessionId, this.node.id]);
    }
    saveSession() {
        const key = this.sessionKey();
        this.finishStroke();
        const state = Object.fromEntries(sessionFields.map(field => [field, this[field]]));
        state.tools = Object.fromEntries(toolFields.map(field => [field, this.$(`.${field}`).value]));
        sessions.set(key, structuredClone(state));
    }
    restoreSession() {
        const saved = sessions.get(this.sessionKey());
        if (!saved) return;
        const state = structuredClone(saved);
        for (const field of sessionFields) this[field] = state[field];
        for (const field of ["width", "height", "fps", "zoom"]) this.$(`.${field}`).value = this[field];
        for (const field of toolFields) this.$(`.${field}`).value = state.tools[field];
        this.$(".range-tools").hidden = this.$(".scope").value !== "range";
        this.gallery(); this.update();
    }
    message(text, error = false) { this.$(".status").textContent = text; this.$(".status").classList.toggle("error", error); }
    selection() { return this.clips.find(c => c.uid === this.selected); }
    duration(clip) { return Math.max(1, Math.round((clip.out - clip.in) * this.fps)) / this.fps; }
    total() { return this.clips.reduce((sum, c) => sum + this.duration(c), 0); }
    locate(time) {
        let start = 0;
        for (const clip of this.clips) {
            const duration = this.duration(clip);
            if (time < start + duration) return { clip, start, local: time - start };
            start += duration;
        }
        return null;
    }
    clipStart(clip) { let start = 0; for (const c of this.clips) { if (c === clip) return start; start += this.duration(c); } return 0; }
    remember() { this.undo.push(structuredClone(this.clips)); if (this.undo.length > 50) this.undo.shift(); this.redo = []; }
    history(redo) {
        const from = redo ? this.redo : this.undo, to = redo ? this.undo : this.redo;
        if (!from.length) return;
        to.push(structuredClone(this.clips)); this.clips = from.pop(); this.update();
    }
    async refresh() {
        if (this.refreshing || this.disposed) return;
        this.refreshing = true;
        try {
            const data = await (await request("/media")).json();
            if (this.disposed) return;
            this.$(".queue").textContent = `${data.running} running · ${data.pending} pending`;
            this.hasFFmpeg = data.ffmpeg;
            const signature = JSON.stringify(data.media);
            if (signature !== this.mediaSignature) {
                this.mediaSignature = signature;
                this.assets = new Map(data.media.map(a => [a.id, a]));
                for (const id of this.metadata.keys()) if (!this.assets.has(id)) this.metadata.delete(id);
                this.gallery(); this.timeline(); this.sync();
            }
            if (!data.ffmpeg) this.message("Install FFmpeg and ffprobe to edit video.", true);
        } catch (error) { if (!this.disposed) this.message(`Could not read the queue: ${error.message}`, true); }
        finally { this.refreshing = false; }
    }
    gallery() {
        const gallery = this.$(".gallery"); gallery.replaceChildren();
        const filter = this.$(".filter").value;
        for (const asset of [...this.assets.values()].reverse()) {
            if (filter !== "all" && asset.kind !== filter) continue;
            const card = document.createElement("button"); card.className = "asset"; card.draggable = true;
            card.title = `${asset.filename}\nNode ${asset.node_id} · ${asset.prompt_id}\nDrag or double-click to add`;
            const preview = document.createElement(asset.kind === "image" ? "img" : "video");
            preview.src = api.apiURL(`${BASE}/source/${asset.id}`);
            if (asset.kind === "video") { preview.muted = true; preview.preload = "metadata"; }
            else { preview.loading = "lazy"; preview.alt = asset.filename; }
            const label = document.createElement("span"); label.textContent = asset.filename;
            card.append(preview, label);
            card.ondragstart = e => { e.dataTransfer.setData(mime, asset.id); e.dataTransfer.effectAllowed = "copy"; };
            card.ondblclick = () => this.add(asset.id).catch(error => this.message(error.message, true));
            gallery.append(card);
        }
        if (!gallery.children.length) {
            const empty = document.createElement("div"); empty.className = "empty";
            empty.textContent = "Results still in ComfyUI's temporary history will appear here."; gallery.append(empty);
        }
    }
    async add(id, index = this.clips.length) {
        const asset = this.assets.get(id); if (!asset) throw new Error("This result is no longer available.");
        this.message("Loading media…");
        let meta = this.metadata.get(id);
        if (!meta) { meta = await (await request(`/media/${id}`)).json(); this.metadata.set(id, meta); }
        if (this.disposed || !this.assets.has(id)) return;
        this.remember();
        const clip = { uid: uid(), media_id: id, name: asset.filename, kind: asset.kind, meta, in: 0, out: meta.duration, strokes: [] };
        this.clips.splice(index, 0, clip); this.selected = clip.uid;
        if (this.clips.length === 1) {
            this.width = Math.ceil(meta.width / 2) * 2; this.height = Math.ceil(meta.height / 2) * 2;
            this.fps = asset.kind === "video" ? Math.min(240, meta.fps) : 24;
            for (const key of ["width", "height", "fps"]) this.$(`.${key}`).value = this[key];
        }
        this.time = this.clipStart(clip); this.update(); this.message("Media added. Select Pencil to draw.");
    }
    trim(clip, key, value) {
        const frame = 1 / this.fps;
        if (!Number.isFinite(value)) return;
        if (key === "in") clip.in = clamp(value, 0, Math.max(0, clip.out - frame));
        else clip.out = clamp(value, clip.in + frame, clip.kind === "video" ? clip.meta.duration : Math.max(value, clip.in + frame));
        for (const stroke of clip.strokes) if (stroke.scope === "clip") { stroke.start = clip.in; stroke.end = clip.out; }
        this.time = this.clipStart(clip);
    }
    timeline() {
        const timeline = this.$(".timeline"); timeline.replaceChildren();
        const total = this.total(), width = Math.max(total * this.pixelsPerSecond + 80, 600);
        timeline.style.width = `${width}px`;
        const ruler = document.createElement("div"); ruler.className = "ruler";
        const interval = this.pixelsPerSecond < 30 ? 5 : this.pixelsPerSecond < 80 ? 2 : 1;
        for (let t = 0; t < width / this.pixelsPerSecond; t += interval) {
            const tick = document.createElement("span"); tick.className = "tick"; tick.style.left = `${t * this.pixelsPerSecond}px`; tick.textContent = `${t}s`; ruler.append(tick);
        }
        ruler.onpointerdown = e => {
            const seek = event => this.seek((event.clientX - timeline.getBoundingClientRect().left) / this.pixelsPerSecond);
            ruler.setPointerCapture(e.pointerId); seek(e); ruler.onpointermove = seek;
            ruler.onpointerup = () => { ruler.onpointermove = null; };
        };
        timeline.append(ruler);
        let offset = 0;
        for (const clip of this.clips) {
            const start = offset, duration = this.duration(clip);
            const block = document.createElement("div"); block.className = `clip ${clip.kind}`; block.draggable = true;
            block.classList.toggle("selected", clip.uid === this.selected); block.classList.toggle("missing", !this.assets.has(clip.media_id));
            block.style.left = `${start * this.pixelsPerSecond}px`; block.style.width = `${duration * this.pixelsPerSecond}px`;
            const label = document.createElement("span"); label.className = "clip-name"; label.textContent = this.assets.has(clip.media_id) ? clip.name : "Unavailable";
            const length = document.createElement("small"); length.textContent = `${duration.toFixed(2)}s`;
            if (this.assets.has(clip.media_id)) {
                const video = clip.kind === "video" || clip.frame_time != null;
                const thumbnail = document.createElement(video ? "video" : "img");
                thumbnail.className = "clip-thumbnail";
                thumbnail.src = api.apiURL(`${BASE}/source/${clip.media_id}`) + (video ? `#t=${clip.frame_time ?? clip.in}` : "");
                if (video) { thumbnail.muted = true; thumbnail.preload = "metadata"; }
                else thumbnail.alt = "";
                block.append(thumbnail);
            }
            block.append(label, length);
            block.onclick = () => { this.selected = clip.uid; this.seek(start); this.timeline(); this.inspector(); };
            block.ondragstart = e => { e.dataTransfer.setData(clipMime, clip.uid); e.dataTransfer.effectAllowed = "move"; };
            for (const key of ["in", "out"]) {
                const handle = document.createElement("div"); handle.className = `handle ${key === "in" ? "left" : "right"}`;
                handle.title = key === "in" ? "Trim start" : "Trim end / duration";
                handle.onpointerdown = e => {
                    e.stopPropagation(); e.preventDefault(); block.draggable = false; handle.setPointerCapture(e.pointerId);
                    this.remember(); this.selected = clip.uid; this.stop(); const origin = e.clientX, initial = clip[key];
                    handle.onpointermove = event => {
                        this.trim(clip, key, initial + (event.clientX - origin) / this.pixelsPerSecond);
                        block.style.width = `${this.duration(clip) * this.pixelsPerSecond}px`; length.textContent = `${this.duration(clip).toFixed(2)}s`; this.inspector();
                    };
                    handle.onpointerup = handle.onpointercancel = () => { handle.onpointermove = null; block.draggable = true; this.update(); };
                };
                block.append(handle);
            }
            timeline.append(block);
            for (const stroke of clip.strokes) {
                const begin = Math.max(clip.in, stroke.start), end = Math.min(clip.out, stroke.end);
                if (end <= begin) continue;
                const ink = document.createElement("div"); ink.className = "ink-span";
                ink.style.left = `${(start + begin - clip.in) * this.pixelsPerSecond}px`; ink.style.width = `${Math.max(2, (end - begin) * this.pixelsPerSecond)}px`; timeline.append(ink);
            }
            offset += duration;
        }
        const head = document.createElement("div"); head.className = "playhead"; head.style.left = `${this.time * this.pixelsPerSecond}px`; timeline.append(head);
    }
    inspector() {
        const clip = this.selection();
        this.$(".trim-in").value = clip?.in.toFixed(3) ?? 0; this.$(".trim-out").value = clip?.out.toFixed(3) ?? 0;
        for (const key of ["in", "out"]) this.$(`.trim-${key}`).disabled = !clip;
    }
    update() {
        this.stop(); this.time = clamp(this.time, 0, Math.max(0, this.total() - 1 / this.fps));
        if (!this.selection()) this.selected = this.locate(this.time)?.clip.uid ?? null;
        this.canvas.width = this.width; this.canvas.height = this.height;
        this.fitCanvas();
        this.$(".scrub").max = Math.max(0, this.total() - 1 / this.fps);
        this.timeline(); this.inspector(); this.sync();
    }
    seek(time) {
        this.stop(); this.time = clamp(time, 0, Math.max(0, this.total() - 1 / this.fps));
        const clip = this.locate(this.time)?.clip;
        if (clip && clip.uid !== this.selected) { this.selected = clip.uid; this.timeline(); this.inspector(); }
        this.sync();
    }
    get pixelsPerSecond() { return this.zoom * (this.uiScale ?? 1); }
    scaleInterface() {
        const scale = Math.round(clamp(Math.sqrt(this.root.clientWidth * this.root.clientHeight / (1120 * 820)), .85, 2) * 100) / 100;
        if (scale === this.uiScale) return;
        this.uiScale = scale;
        this.root.style.setProperty("--ui-scale", scale);
        this.timeline();
        this.fitCanvas();
    }
    fitCanvas() {
        const stage = this.$(".stage");
        const width = Math.min(stage.clientWidth, stage.clientHeight * this.width / this.height);
        this.canvas.style.width = `${width}px`; this.canvas.style.height = `${width * this.height / this.width}px`;
    }
    stop() { this.playing = false; this.video.pause(); this.$('[data-action="play"]').textContent = "▶"; }
    togglePlay() {
        if (this.playing) return this.stop();
        if (!this.clips.length) return;
        if (this.time >= this.total() - 1 / this.fps) this.time = 0;
        this.drawing = false; this.$('[data-action="pencil"]').classList.remove("active"); this.$(".stage").classList.remove("drawing");
        this.playing = true; this.previousTick = performance.now(); this.$('[data-action="play"]').textContent = "❚❚"; this.sync();
    }
    clearSource() {
        ++this.loadVersion; this.loaded = null; this.ready = false;
        this.video.pause(); this.video.removeAttribute("src"); this.video.load();
        this.picture = new Image();
        if (this.frameUrl) { URL.revokeObjectURL(this.frameUrl); this.frameUrl = null; }
    }
    async load(clip) {
        this.clearSource(); this.loaded = clip.uid; const version = this.loadVersion;
        try {
            let url = api.apiURL(`${BASE}/source/${clip.media_id}`);
            if (clip.frame_time != null) {
                const response = await request("/frame", jsonOptions({ media_id: clip.media_id, time: clip.frame_time }));
                const blob = await response.blob(); if (version !== this.loadVersion || this.disposed) return;
                this.frameUrl = URL.createObjectURL(blob); url = this.frameUrl;
            }
            if (clip.kind === "video") {
                this.video.src = url;
                await new Promise((resolve, reject) => {
                    this.video.onloadedmetadata = resolve;
                    this.video.onerror = () => reject(new Error("Video format not supported by the browser."));
                });
            } else {
                this.picture = new Image(); this.picture.src = url; await this.picture.decode();
            }
            if (version !== this.loadVersion || this.disposed) return;
            this.ready = true; this.sync();
        } catch (error) { if (version === this.loadVersion && !this.disposed) this.message(error.message, true); }
    }
    sync() {
        const current = this.locate(this.time);
        this.$(".scrub").value = this.time;
        this.$(".time").textContent = `${seconds(this.time)} / ${seconds(this.total())}`;
        const head = this.$(".playhead"); if (head) head.style.left = `${this.time * this.pixelsPerSecond}px`;
        if (!current || !this.assets.has(current.clip.media_id)) {
            if (this.loaded) this.clearSource();
            this.stop(); this.paint(); return;
        }
        if (this.loaded !== current.clip.uid) { this.load(current.clip); this.paint(); return; }
        if (this.ready && current.clip.kind === "video") {
            const target = Math.min(current.clip.out - .001, current.clip.in + current.local);
            if (Math.abs(this.video.currentTime - target) > (this.playing ? .2 : .001) && !this.video.seeking) this.video.currentTime = target;
            if (this.playing && this.video.paused) this.video.play().catch(error => { this.stop(); this.message(error.message, true); });
        }
        this.paint();
    }
    rect(clip) {
        const scale = Math.min(this.width / clip.meta.width, this.height / clip.meta.height);
        const w = clip.meta.width * scale, h = clip.meta.height * scale;
        return { x: (this.width - w) / 2, y: (this.height - h) / 2, w, h };
    }
    drawStroke(ctx, stroke, rect) {
        ctx.save(); ctx.translate(rect.x, rect.y);
        ctx.beginPath(); ctx.rect(0, 0, rect.w, rect.h); ctx.clip();
        ctx.strokeStyle = stroke.color; ctx.fillStyle = stroke.color; ctx.globalAlpha = stroke.opacity;
        ctx.lineWidth = stroke.width * rect.w; ctx.lineCap = "round"; ctx.lineJoin = "round";
        ctx.beginPath();
        stroke.points.forEach(([x, y], i) => { if (i) ctx.lineTo(x * rect.w, y * rect.h); else ctx.moveTo(x * rect.w, y * rect.h); });
        if (stroke.points.length === 1) { const [x, y] = stroke.points[0]; ctx.arc(x * rect.w, y * rect.h, ctx.lineWidth / 2, 0, Math.PI * 2); ctx.fill(); }
        else ctx.stroke();
        ctx.restore();
    }
    paint() {
        const ctx = this.ctx, current = this.locate(this.time);
        ctx.fillStyle = "#080d12"; ctx.fillRect(0, 0, this.width, this.height);
        this.$(".hint").hidden = !!current;
        if (!current) this.$(".hint").textContent = "Drag an image or video onto the timeline.";
        if (!current) return;
        const { clip, local } = current;
        if (!this.assets.has(clip.media_id)) { this.$(".hint").hidden = false; this.$(".hint").textContent = "This media is no longer in temporary history."; return; }
        if (!this.ready || this.loaded !== clip.uid) return;
        const rect = this.rect(clip), source = clip.kind === "video" ? this.video : this.picture;
        if (clip.kind === "video" && this.video.readyState < 2) return;
        ctx.drawImage(source, rect.x, rect.y, rect.w, rect.h);
        for (const stroke of clip.strokes) if (stroke.start <= clip.in + local && clip.in + local < stroke.end) this.drawStroke(ctx, stroke, rect);
        if (this.stroke) this.drawStroke(ctx, this.stroke, rect);
    }
    point(event, clip) {
        const bounds = this.canvas.getBoundingClientRect(), rect = this.rect(clip);
        return [clamp(((event.clientX - bounds.left) * this.width / bounds.width - rect.x) / rect.w, 0, 1),
                clamp(((event.clientY - bounds.top) * this.height / bounds.height - rect.y) / rect.h, 0, 1)];
    }
    startStroke(event) {
        if (!this.drawing || event.button !== 0 || !this.ready) return;
        const current = this.locate(this.time); if (!current || !this.assets.has(current.clip.media_id)) return;
        this.stop(); this.selected = current.clip.uid;
        const clip = current.clip, scope = this.$(".scope").value;
        let start = clip.in, end = clip.out;
        if (scope === "frame") { start = clip.in + Math.floor(current.local * this.fps + 1e-6) / this.fps; end = Math.min(clip.out, start + 1 / this.fps); }
        if (scope === "range") { start = clip.in + Number(this.$(".ink-in").value); end = Math.min(clip.out, clip.in + Number(this.$(".ink-out").value)); }
        if (!Number.isFinite(start) || !Number.isFinite(end) || start < clip.in || end <= start) return this.message("The drawing range is invalid.", true);
        this.remember(); this.strokeClip = clip;
        this.stroke = { color: this.$(".color").value, width: clamp(Number(this.$(".brush").value) || 1, 1, 200) / clip.meta.width,
            opacity: Number(this.$(".opacity").value), scope, start, end, points: [this.point(event, clip)] };
        this.canvas.setPointerCapture(event.pointerId); this.paint();
    }
    moveStroke(event) { if (this.stroke) { this.stroke.points.push(this.point(event, this.strokeClip)); this.paint(); } }
    finishStroke() {
        if (!this.stroke) return;
        this.strokeClip.strokes.push(this.stroke); this.stroke = null; this.strokeClip = null; this.timeline(); this.paint();
    }
    tick(now) {
        if (this.disposed) return;
        const dt = Math.min(.1, (now - this.previousTick) / 1000); this.previousTick = now;
        if (this.playing) {
            const current = this.locate(this.time);
            if (this.ready && current && (current.clip.kind === "image" || (!this.video.seeking && this.video.readyState >= 2))) this.time += dt;
            if (this.time >= this.total()) { this.time = Math.max(0, this.total() - 1 / this.fps); this.stop(); }
            this.sync();
        }
        this.animation = requestAnimationFrame(time => this.tick(time));
    }
    async png(extract) {
        this.stop(); const current = this.locate(this.time);
        if (!current) throw new Error("Select an image or frame.");
        const { clip, local } = current;
        const sourceTime = clip.frame_time ?? (clip.in + local);
        const strokes = clip.frame_time == null ? clip.strokes : clip.strokes.filter(s => s.start <= clip.in + local && clip.in + local < s.end).map(s => ({ ...s, start: 0, end: sourceTime + 1 }));
        const response = await request("/frame", jsonOptions({ media_id: clip.media_id, time: sourceTime, strokes: extract ? [] : strokes }));
        if (!extract) { download(await response.blob(), "image-editada.png"); this.message("PNG downloaded at original resolution."); return; }
        await response.arrayBuffer();
        this.remember();
        const extracted = { uid: uid(), media_id: clip.media_id, name: `Frame · ${clip.name}`, kind: "image", meta: { ...clip.meta, duration: 3, audio: false },
            frame_time: sourceTime, in: 0, out: 3, strokes: clip.strokes.filter(s => s.start <= clip.in + local && clip.in + local < s.end).map(s => ({ ...structuredClone(s), start: 0, end: 3 })) };
        this.clips.splice(this.clips.indexOf(clip) + 1, 0, extracted); this.selected = extracted.uid;
        this.time = this.clipStart(extracted); this.update(); this.message("Frame added to the timeline as an image.");
    }
    async export() {
        if (this.job) return;
        if (!this.clips.length) throw new Error("The timeline is empty.");
        if (this.clips.some(c => !this.assets.has(c.media_id))) throw new Error("Some media is no longer in history.");
        this.stop(); this.job = uid();
        this.$("progress").hidden = false; this.$("progress").value = 0;
        this.$('[data-action="cancel"]').hidden = false; this.$('[data-action="export"]').disabled = true;
        this.message("Preparing export…");
        try {
            const response = await request(`/export/${this.job}`, jsonOptions({ client_id: api.clientId, width: this.width, height: this.height, fps: this.fps, clips: this.clips }));
            const blob = await response.blob();
            if (!this.disposed && !this.cancelled) { download(blob, "montaje.mp4"); this.message("MP4 downloaded."); }
        } finally {
            this.job = null; this.cancelled = false;
            this.$("progress").hidden = true; this.$('[data-action="cancel"]').hidden = true; this.$('[data-action="export"]').disabled = false;
        }
    }
    async cancel() {
        if (!this.job) return;
        this.cancelled = true; await request(`/export/${this.job}`, { method: "DELETE" }); this.message("Export cancelled.");
    }
    async act(action) {
        const clip = this.selection();
        switch (action) {
            case "fullscreen": if (document.fullscreenElement) await document.exitFullscreen(); else await this.root.requestFullscreen(); break;
            case "refresh": await this.refresh(); break;
            case "pencil": this.stop(); this.drawing = !this.drawing; this.$('[data-action="pencil"]').classList.toggle("active", this.drawing); this.$(".stage").classList.toggle("drawing", this.drawing); break;
            case "undo": this.history(false); break;
            case "redo": this.history(true); break;
            case "play": this.togglePlay(); break;
            case "previous": this.seek(this.time - 1 / this.fps); break;
            case "next": this.seek(this.time + 1 / this.fps); break;
            case "clearInk": if (clip) { this.remember(); clip.strokes = []; this.update(); } break;
            case "remove": if (clip) { this.remember(); this.clips.splice(this.clips.indexOf(clip), 1); this.update(); } break;
            case "duplicate": if (clip) { this.remember(); const copy = { ...structuredClone(clip), uid: uid() }; this.clips.splice(this.clips.indexOf(clip) + 1, 0, copy); this.selected = copy.uid; this.update(); } break;
            case "split": {
                const current = this.locate(this.time); if (!current) return;
                const cut = current.clip.in + Math.round(current.local * this.fps) / this.fps;
                if (cut <= current.clip.in || cut >= current.clip.out) return;
                this.remember(); const tail = { ...structuredClone(current.clip), uid: uid(), in: cut };
                current.clip.out = cut; this.clips.splice(this.clips.indexOf(current.clip) + 1, 0, tail); this.selected = tail.uid; this.update(); break;
            }
            case "png": await this.png(false); break;
            case "extract": await this.png(true); break;
            case "export": await this.export(); break;
            case "cancel": await this.cancel(); break;
        }
    }
    dispose() {
        this.saveSession();
        this.disposed = true; clearInterval(this.timer); clearTimeout(this.refreshDelay); cancelAnimationFrame(this.animation);
        this.resizeObserver.disconnect(); this.layoutObserver.disconnect();
        api.removeEventListener("executed", this.onExecuted); api.removeEventListener("yafv-editor-progress", this.onProgress);
        this.cancel().catch(() => {}); this.clearSource();
        this.clips = []; this.undo = []; this.redo = []; this.assets.clear(); this.metadata.clear();
    }
}

app.registerExtension({
    name: "YAFV.MediaEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "YAFVMediaEditor") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args); this.mediaEditor = new Editor(this); return result;
        };
        const serialized = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            const result = serialized?.apply(this, arguments);
            this.mediaEditor.saveSession();
            data.yafv_editor_session = this.mediaEditor.sessionId;
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (data) {
            const result = configured?.apply(this, arguments);
            if (data.yafv_editor_session) this.mediaEditor.sessionId = data.yafv_editor_session;
            this.mediaEditor.restoreSession();
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) { this.mediaEditor?.dispose(); return removed?.apply(this, args); };
    },
});
