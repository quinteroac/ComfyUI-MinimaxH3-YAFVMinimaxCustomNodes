import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { PromptMediaPicker } from "./prompt_media_picker.js";

const stylesheet = document.createElement("link");
stylesheet.rel = "stylesheet";
stylesheet.href = new URL("./video_prompts.css", import.meta.url).href;
document.head.append(stylesheet);
const panels = new Set();
const BASE = "/yafv/prompts";
const settingDefinitions = [
    ["max_length", "Maximum length", "number", 1, 32768, 1],
    ["sampling_mode", "Sampling", ["off", "on"]],
    ["temperature", "Temperature", "number", .01, 2, .01],
    ["top_k", "Top K", "number", 0, 1000, 1],
    ["top_p", "Top P", "number", 0, 1, .01],
    ["min_p", "Min P", "number", 0, 1, .01],
    ["repetition_penalty", "Repetition penalty", "number", 0, 5, .01],
    ["presence_penalty", "Presence penalty", "number", 0, 5, .01],
    ["seed", "Seed", "number", 0, Number.MAX_SAFE_INTEGER, 1],
    ["thinking", "Thinking", "checkbox"],
    ["use_default_template", "Native template", "checkbox"],
    ["mtp", "MTP", ["auto", "off", "2", "3", "4", "5"]],
];

async function request(path, options) {
    const response = await api.fetchApi(BASE + path, options);
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${response.status}`);
    }
    return response;
}
const post = data => ({method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});

class PromptPanel {
    constructor(node, referenceMode = false) {
        this.referenceMode = referenceMode;
        this.media = referenceMode ? [
            ...Array.from({length: 8}, (_, i) => [`ref_image_${i}`, "image", `ref_image_${i}`]),
            ...Array.from({length: 2}, (_, i) => [
                [`ref_video_${i}`, "video", `ref_video_${i} · 24 fps`],
                [`ref_video_audio_${i}`, "audio", `ref_video_audio_${i} · Video audio`],
            ]).flat(),
            ...Array.from({length: 3}, (_, i) => [`ref_audio_${i}`, "audio", `ref_audio_${i}`]),
        ] : [["first", "image", "First frame"], ["last", "image", "Last frame"]];
        this.node = node; this.collection = null; this.entries = []; this.current = null;
        this.dirty = false; this.drafting = false; this.busy = false; this.disposed = false; this.hydrated = false;
        this.files = {}; this.imageActions = {}; this.urls = {};
        this.root = document.createElement("div"); this.root.className = "yafv-prompts";
        this.root.classList.toggle("reference-prompts", referenceMode);
        this.root.innerHTML = `
          <header><strong>${referenceMode ? "Reference to Video Prompts" : "Video Prompts"}</strong><span class="badge">ComfyUI session</span></header>
          <div class="unsaved" hidden><p>You have unsaved changes. What would you like to do before continuing?</p><div class="actions"><button data-action="saveContinue" class="primary">Save and continue</button><button data-action="discard">Discard</button><button data-action="stay">Keep editing</button></div></div>
          <div class="body"><aside><div class="actions"><b>My prompts</b><span class="count muted">0</span><button data-action="new">+ New</button></div><div class="entries"></div></aside>
          <section class="form"><div class="reference-toolbar"><b>${referenceMode ? "References" : "Frames"}</b><span class="reference-count muted"></span></div><p class="context-note" hidden></p><div class="frames" aria-label="${referenceMode ? "Reference media" : "Start and end frames"}"></div>
            <label>Item prompt</label><textarea class="prompt" placeholder="Describe the scene, movement, or action…" aria-label="Item prompt"></textarea>
            <div class="actions"><button class="primary" data-action="save">Add</button><button class="danger" data-action="delete">Delete item</button><span class="draft muted"></span></div>
            <details open><summary>Last output prompt</summary><textarea class="generated" readonly placeholder="The result will appear after running the workflow." aria-label="Last output prompt"></textarea></details>
          </section></div>
          <details class="advanced"><summary>Generate Text options</summary><div class="settings"></div><p class="muted">Uses the connected model's capabilities. ${referenceMode ? "Images and up to eight frames per video require vision support. Audio is not analyzed. Loading another video replaces its audio track; you can replace or remove it." : "Frames require vision support. Instructions are included in the text sent to the model."}</p></details>
          <footer><span class="selection"></span><span class="status" role="status">Loading list…</span><button data-action="refresh">Refresh</button></footer>`;
        this.$ = selector => this.root.querySelector(selector);
        this.root.addEventListener("pointerdown", event => event.stopPropagation());
        this.root.addEventListener("wheel", event => event.stopPropagation());
        this.root.addEventListener("keydown", event => event.stopPropagation());
        this.root.addEventListener("click", event => {
            const button = event.target.closest("[data-action]");
            if (!button) return;
            const {action, id} = button.dataset;
            if (action === "addReference") this.chooseMedia(null, id);
            else if (action === "chooseMedia" || action === "replaceDetail") {
                const name = id || this.detailName;
                this.chooseMedia(name, this.media.find(([slot]) => slot === name)[1]);
            } else this.run(() => this.action(action, id));
        });
        this.$(".prompt").oninput = () => this.markDirty();
        this.picker = new PromptMediaPicker(this.root);
        this.detail = document.createElement("dialog"); this.detail.className = "media-detail";
        this.detail.innerHTML = `<div class="dialog-heading"><strong></strong><button type="button" class="detail-close" aria-label="Close preview">×</button></div><div class="detail-preview"></div><p class="detail-info muted"></p><div class="detail-audio"></div><div class="dialog-actions"><button data-action="replaceDetail">Replace</button><button data-action="removeDetail" class="danger">Remove</button></div>`;
        this.root.append(this.detail);
        this.detail.querySelector(".detail-close").onclick = () => this.closeDetail();
        this.detail.addEventListener("cancel", event => { event.preventDefault(); this.closeDetail(); });
        this.detail.addEventListener("click", event => { if (event.target === this.detail) this.closeDetail(); });
        this.renderMedia();
        for (const definition of settingDefinitions) this.settingControl(definition);
        for (const widget of node.widgets ?? []) {
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        }
        this.widget = node.addDOMWidget("video_prompt_panel", "yafv_video_prompts", this.root, {
            serialize: false, getMinHeight: () => 580,
        });
        this.widget.serialize = false;
        this.resizeObserver = new ResizeObserver(() => {
            const scale = Math.max(.85, Math.min(1.7, Math.sqrt(this.root.clientWidth * this.root.clientHeight / (1000 * 760))));
            this.root.style.setProperty("--unit", `${scale.toFixed(2)}px`);
        });
        this.resizeObserver.observe(this.root);
        this.onExecuted = ({detail}) => {
            const output = detail.output?.yafv_prompt?.[0];
            if (output?.collection !== this.collection) return;
            if (output.revision === this.current?.revision) {
                this.$(".generated").value = output.text;
                this.message("Prompt generated. Outputs are ready.");
            }
            this.refresh();
        };
        api.addEventListener("executed", this.onExecuted);
        this.timer = setInterval(() => this.refresh(), 3000);
        node.setSize([1020, 840]);
        panels.add(this); this.controls();
        queueMicrotask(() => this.activate());
    }
    value(name, value) {
        const widget = this.node.widgets.find(w => w.name === name);
        if (!widget) return undefined;
        if (arguments.length === 2 && widget.value !== value) {
            widget.value = value; widget.callback?.(value);
            this.node.setDirtyCanvas?.(true, true);
        }
        return widget.value;
    }
    message(text, error = false) { this.$(".status").textContent = text; this.$(".status").classList.toggle("error", error); }
    async run(action) {
        if (this.busy || this.disposed) return;
        this.busy = true; this.controls();
        try { await action(); }
        catch (error) { if (!this.disposed) this.message(error.message, true); }
        finally { this.busy = false; if (!this.disposed) this.controls(); }
    }
    controls() {
        for (const button of this.root.querySelectorAll("button")) {
            if (!button.closest(".media-picker")) button.disabled = this.busy || !this.hydrated;
        }
        this.$('[data-action="refresh"]').disabled = this.busy;
        this.$('[data-action="save"]').textContent = this.current ? "Save changes" : "Add";
        this.$('[data-action="save"]').disabled ||= !this.$(".prompt").value.trim();
        this.$('[data-action="delete"]').disabled ||= !this.current;
        this.$(".prompt").disabled = this.busy || !this.hydrated;
        const selected = this.entries.find(e => e.revision === this.value("revision_id"));
        this.$(".selection").textContent = selected ? `Will run: ${selected.prompt.slice(0, 65)}` : "No item selected";
        this.$(".draft").textContent = this.dirty ? "Unsaved changes: the queue uses the saved item." : "";
        if (this.referenceMode) {
            const counts = ["image", "video", "audio"].map(kind => {
                const slots = this.referenceSlots(kind), count = slots.filter(([name]) => this.hasMedia(name)).length;
                const add = this.$(`[data-action="addReference"][data-id="${kind}"]`);
                if (add) add.disabled ||= count === slots.length;
                return `${count}/${slots.length} ${{image: "images", video: "videos", audio: "audios"}[kind]}`;
            });
            this.$(".reference-count").textContent = counts.join(" · ");
        }
        for (const button of this.root.querySelectorAll('[data-action="previewMedia"], [data-action="removeImage"]')) button.disabled ||= !this.hasMedia(button.dataset.id);
        this.mode();
    }
    mode() {
        const connected = name => this.node.inputs?.find(input => input.name === name)?.link != null;
        const note = this.$(".context-note");
        note.hidden = !connected(this.referenceMode ? "context_video" : "context_image");
        note.textContent = this.referenceMode ? "Connected context video is used for prompt generation; reference outputs remain as selected." : "Connected context image replaces the Start frame when running the workflow.";
        this.$(".first")?.classList.toggle("context-override", connected("context_image"));
        this.$(".badge").textContent = connected("clip") && connected("text") ? "CLIP + instructions · generates when text is not empty" : "Original text";
    }
    async activate() {
        if (this.disposed || !this.node.graph || this.node.id == null || this.node.id === -1) return;
        const key = JSON.stringify(this.referenceMode ? ["reference", this.node.graph.id, this.node.id] : [this.node.graph.id, this.node.id]);
        if (this.collection === key && this.hydrated) return;
        this.collection = key; this.value("collection_id", key); this.value("revision_id", "");
        this.settingsFromWidgets(); await this.refresh(true);
    }
    async refresh(force = false) {
        if (this.disposed || !this.collection || this.loading || (this.busy && !force)) return;
        const key = this.collection; this.loading = true;
        try {
            const data = await (await request(`/library?${new URLSearchParams({collection: key})}`)).json();
            if (this.disposed || key !== this.collection) return;
            const restarted = this.epoch && this.epoch !== data.epoch;
            this.apply(data, force || !this.hydrated || restarted);
            if (restarted) this.message("ComfyUI restarted: the temporary list is empty.");
        } catch (error) { if (!this.disposed) this.message(`Could not read the list: ${error.message}`, true); }
        finally { this.loading = false; if (!this.disposed) this.controls(); }
    }
    apply(data, replaceForm) {
        this.epoch = data.epoch; this.entries = data.entries; this.hydrated = true;
        const selected = this.entries.find(entry => entry.id === data.selected) ?? null;
        this.value("revision_id", selected?.revision ?? "");
        if (replaceForm || (!this.dirty && !this.drafting && this.current?.revision !== selected?.revision)) this.show(selected);
        else if (selected && this.current?.revision === selected.revision) this.$(".generated").value = selected.generated;
        this.renderList(); this.controls();
    }
    renderList() {
        const list = this.$(".entries"); list.replaceChildren(); this.$(".count").textContent = this.entries.length;
        for (const entry of this.entries) {
            const row = document.createElement("div"); row.className = "entry";
            row.classList.toggle("selected", entry.revision === this.value("revision_id"));
            const choose = document.createElement("button"); choose.className = "choose"; choose.dataset.action = "select"; choose.dataset.id = entry.id;
            const title = document.createElement("span"); title.className = "title"; title.textContent = entry.prompt;
            const frames = document.createElement("small"); frames.textContent = this.referenceMode
                ? ["image", "video", "audio"].map(kind => {
                    const count = this.referenceSlots(kind).filter(([name]) => entry[name]).length;
                    return count ? `${count} ${kind}${count === 1 ? "" : "s"}` : "";
                }).filter(Boolean).join(" · ") || "No references"
                : `${entry.first ? "● Inicio" : "○ Inicio"} · ${entry.last ? "● Final" : "○ Final"}`;
            choose.append(title, frames);
            const remove = document.createElement("button"); remove.className = "remove danger"; remove.textContent = "×";
            remove.title = "Delete item"; remove.setAttribute("aria-label", "Delete item"); remove.dataset.action = "delete"; remove.dataset.id = entry.id;
            row.append(choose, remove); list.append(row);
        }
        if (!this.entries.length) { const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = this.referenceMode ? "Add a prompt and optional references. The list lasts until ComfyUI restarts." : "Add a prompt and optional first and last frames. The list lasts until ComfyUI restarts."; list.append(empty); }
    }
    markDirty() { this.dirty = true; this.controls(); }
    releaseURLs() {
        for (const url of Object.values(this.urls)) URL.revokeObjectURL(url);
        this.urls = {};
    }
    show(entry) {
        this.releaseURLs(); this.files = {}; this.imageActions = {}; this.current = entry; this.dirty = false; this.drafting = false;
        this.$(".prompt").value = entry?.prompt ?? "";
        this.$(".generated").value = entry?.generated ?? "";
        this.closeDetail();
        this.renderMedia();
        this.message(entry ? "Item selected. Run the workflow from the ComfyUI queue." : "Write a prompt and click Add.");
        this.controls();
    }
    referenceSlots(kind) {
        return this.media.filter(([name, type]) => type === kind && !name.startsWith("ref_video_audio_"));
    }
    availableSlot(kind) {
        return this.referenceSlots(kind).find(([name]) => !this.hasMedia(name));
    }
    hasMedia(name) {
        if (this.imageActions[name] === "remove") return false;
        if (this.files[name]) return true;
        if (name.startsWith("ref_video_audio_")) {
            const video = name.replace("ref_video_audio_", "ref_video_");
            if (this.imageActions[video] === "remove") return false;
            if (this.imageActions[video] === "upload") return true;
        }
        return !!this.current?.[name];
    }
    clearMedia(container) {
        for (const element of container.querySelectorAll("video, audio")) { element.pause(); element.removeAttribute("src"); element.load(); }
    }
    renderMedia() {
        const frames = this.$(".frames"); this.clearMedia(frames); frames.replaceChildren();
        for (const [name, kind] of this.media) {
            if (name.startsWith("ref_video_audio_")) continue;
            if (!this.referenceMode || this.hasMedia(name)) this.frameControl(name, kind);
        }
        if (this.referenceMode) {
            const add = document.createElement("div"); add.className = "media-add";
            const title = document.createElement("span"); title.textContent = "+ Add reference"; add.append(title);
            for (const kind of ["image", "video", "audio"]) {
                const button = document.createElement("button"); button.type = "button";
                button.dataset.action = "addReference"; button.dataset.id = kind;
                button.textContent = {image: "Image", video: "Video", audio: "Audio"}[kind]; add.append(button);
            }
            add.ondragover = event => { event.preventDefault(); event.stopPropagation(); add.classList.add("drag"); };
            add.ondragleave = () => add.classList.remove("drag");
            add.ondrop = event => {
                event.preventDefault(); event.stopPropagation(); add.classList.remove("drag");
                this.addDroppedFiles([...event.dataTransfer.files]);
            };
            frames.append(add);
        }
    }
    mediaLabel(name) {
        if (name === "first") return "Start";
        if (name === "last") return "End";
        if (name.startsWith("ref_video_audio_")) return "Video soundtrack";
        const kind = this.media.find(([slot]) => slot === name)?.[1];
        const present = this.referenceSlots(kind).filter(([slot]) => this.hasMedia(slot));
        return `${{image: "Image", video: "Video", audio: "Audio"}[kind]} ${present.findIndex(([slot]) => slot === name) + 1}`;
    }
    mediaTag(name) {
        if (!this.referenceMode) return "";
        // Match reference_sheet: video soundtracks precede standalone audio tags.
        if (name.startsWith("ref_audio_") || name.startsWith("ref_video_audio_")) {
            const videos = this.referenceSlots("video").filter(([slot]) => this.hasMedia(slot));
            if (videos.some(([slot]) => this.imageActions[slot] === "upload" && !this.imageActions[slot.replace("ref_video_", "ref_video_audio_")])) return "Audio tag after save";
            const soundtracks = videos.map(([slot]) => slot.replace("ref_video_", "ref_video_audio_")).filter(slot => this.hasMedia(slot));
            if (name.startsWith("ref_video_audio_")) return `<Audio ${soundtracks.indexOf(name) + 1}>`;
            const index = this.referenceSlots("audio").filter(([slot]) => this.hasMedia(slot)).findIndex(([slot]) => slot === name);
            return `<Audio ${soundtracks.length + index + 1}>`;
        }
        return `<${this.mediaLabel(name).replace("Image ", "Picture ")}>`;
    }
    mediaURL(name) {
        if (!this.hasMedia(name)) return null;
        if (this.urls[name]) return this.urls[name];
        if (name.startsWith("ref_video_audio_")) {
            const video = name.replace("ref_video_audio_", "ref_video_");
            if (this.imageActions[video] === "upload" && !this.imageActions[name]) return this.urls[video];
        }
        return this.current?.[name] ? api.apiURL(`${BASE}/${this.referenceMode ? "media" : "image"}/${this.current.revision}/${name}?${new URLSearchParams({collection: this.collection})}`) : null;
    }
    fileName(name) {
        if (this.files[name]) return this.files[name].name;
        if (name.startsWith("ref_video_audio_")) {
            const video = name.replace("ref_video_audio_", "ref_video_");
            if (this.imageActions[video] === "upload" && !this.imageActions[name]) return `${this.files[video].name} · soundtrack`;
        }
        return this.current?.media_names?.[name] || this.mediaLabel(name);
    }
    setFile(frame, kind, file, render = true) {
        if (!file || this.busy || !this.hydrated || this.disposed) return;
        if (file.type && !file.type.startsWith(`${kind}/`) && file.type !== "application/octet-stream") return this.message(`Select a ${kind} file.`, true);
        if (kind === "video") {
            const audio = frame.replace("ref_video_", "ref_video_audio_");
            if (this.urls[audio]) URL.revokeObjectURL(this.urls[audio]);
            delete this.urls[audio]; delete this.files[audio]; delete this.imageActions[audio];
        }
        if (this.urls[frame]) URL.revokeObjectURL(this.urls[frame]);
        this.files[frame] = file; this.imageActions[frame] = "upload"; this.urls[frame] = URL.createObjectURL(file);
        if (render) this.renderMedia();
        this.markDirty();
    }
    addDroppedFiles(files) {
        if (this.busy || !this.hydrated) return;
        const classified = files.map(file => ({file, kind: file.type.split("/")[0]}));
        if (classified.some(({kind}) => !["image", "video", "audio"].includes(kind))) return this.message("Use the media selector for files with an unknown type.", true);
        for (const kind of ["image", "video", "audio"]) {
            const available = this.referenceSlots(kind).filter(([name]) => !this.hasMedia(name)).length;
            if (classified.filter(item => item.kind === kind).length > available) return this.message(`Only ${available} ${kind} slots are available.`, true);
        }
        for (const {file, kind} of classified) this.setFile(this.availableSlot(kind)[0], kind, file, false);
        this.renderMedia(); this.controls();
    }
    async chooseMedia(name, kind) {
        if (this.busy || !this.hydrated) return;
        const count = name ? 1 : this.referenceSlots(kind).filter(([slot]) => !this.hasMedia(slot)).length;
        if (!count) return;
        const returnToDetail = this.detailName;
        this.closeDetail();
        this.busy = true; this.controls();
        try {
            const files = await this.picker.open(kind, kind === "image" ? count : 1, !!name && this.hasMedia(name));
            this.busy = false;
            if (this.disposed) return;
            if (!files) {
                if (returnToDetail) this.preview(returnToDetail);
                return;
            }
            for (const file of files) this.setFile(name || this.availableSlot(kind)[0], kind, file, false);
            this.renderMedia();
            if (returnToDetail) this.preview(returnToDetail);
        } catch (error) { this.message(error.message, true); }
        finally { this.busy = false; if (!this.disposed) this.controls(); }
    }
    frameControl(name, kind) {
        const section = document.createElement("div"); section.className = `frame ${name}`;
        const label = this.mediaLabel(name), url = this.mediaURL(name);
        section.innerHTML = `<button type="button" class="media-tile" data-action="chooseMedia" data-id="${name}"><span class="tile-empty">+<small>Select ${kind}</small></span></button><div class="tile-actions"><button type="button" data-action="previewMedia" data-id="${name}" aria-label="Preview ${label}" title="Preview">⤢</button><button type="button" data-action="removeImage" data-id="${name}" aria-label="Remove ${label}" title="Remove">×</button></div><div class="tile-caption"><b></b><small></small></div>`;
        section.querySelector("b").textContent = label;
        const caption = section.querySelector(".tile-caption small"); caption.textContent = url ? this.mediaTag(name) : "Optional";
        const tile = section.querySelector(".media-tile"); tile.title = `${url ? this.fileName(name) + " · Click to replace" : "Select " + label} (${name})`;
        tile.setAttribute("aria-label", `${url ? "Replace" : "Select"} ${label}`);
        if (url) {
            tile.replaceChildren();
            if (kind === "audio") { const icon = document.createElement("span"); icon.className = "audio-icon"; icon.textContent = "♪"; tile.append(icon); }
            else {
                const media = document.createElement(kind === "image" ? "img" : "video");
                media.className = "media-preview"; media.src = url; media.draggable = false;
                if (kind === "image") { media.alt = label; media.loading = "lazy"; }
                else { media.muted = true; media.preload = "metadata"; media.playsInline = true; }
                const loaded = () => {
                    const dimensions = `${media.naturalWidth || media.videoWidth} × ${media.naturalHeight || media.videoHeight}`;
                    tile.title = `${this.fileName(name)} · ${dimensions} · Click to replace (${name})`;
                    if (kind === "video" && Number.isFinite(media.duration)) media.currentTime = Math.min(.1, media.duration / 2);
                };
                media.addEventListener(kind === "image" ? "load" : "loadedmetadata", loaded);
                media.addEventListener("error", () => {
                    media.hidden = true;
                    const fallback = document.createElement("small"); fallback.textContent = "Preview unavailable"; tile.append(fallback);
                }, {once: true});
                tile.append(media);
                if (kind === "video") { const play = document.createElement("span"); play.className = "tile-play"; play.textContent = "▶"; tile.append(play); }
            }
        }
        if (kind === "video") {
            const audio = name.replace("ref_video_", "ref_video_audio_");
            const indicator = document.createElement("button"); indicator.type = "button"; indicator.className = "soundtrack-badge";
            indicator.dataset.action = "previewMedia"; indicator.dataset.id = name;
            indicator.textContent = this.imageActions[name] === "upload" && !this.imageActions[audio] ? "♪ Audio: check on save" : this.hasMedia(audio) ? "♪ Audio attached" : "+ Attach audio";
            section.append(indicator);
        }
        tile.ondragover = event => { event.preventDefault(); event.stopPropagation(); tile.classList.add("drag"); };
        tile.ondragleave = () => tile.classList.remove("drag");
        tile.ondrop = event => {
            event.preventDefault(); event.stopPropagation(); tile.classList.remove("drag");
            if (event.dataTransfer.files.length !== 1) return this.message("Drop one file to replace this media.", true);
            this.setFile(name, kind, event.dataTransfer.files[0]);
        };
        this.$(".frames").append(section);
    }
    closeDetail() {
        if (!this.detail) return;
        this.clearMedia(this.detail); this.detail.querySelector(".detail-preview").replaceChildren();
        this.detail.querySelector(".detail-audio").replaceChildren(); this.detail.close(); this.detailName = null;
    }
    preview(name) {
        const kind = this.media.find(([slot]) => slot === name)?.[1], url = this.mediaURL(name);
        if (!url) return;
        this.closeDetail(); this.detailName = name;
        this.detail.querySelector(".dialog-heading strong").textContent = this.mediaLabel(name);
        this.detail.setAttribute("aria-label", `${this.mediaLabel(name)} preview`);
        const info = this.detail.querySelector(".detail-info"); info.textContent = this.fileName(name);
        const media = document.createElement(kind === "image" ? "img" : kind); media.src = url;
        if (kind === "image") media.alt = this.mediaLabel(name);
        else { media.controls = true; media.preload = "metadata"; if (kind === "video") media.muted = true; }
        media.addEventListener(kind === "image" ? "load" : "loadedmetadata", () => {
            const dimensions = kind === "audio" ? "" : `${media.naturalWidth || media.videoWidth} × ${media.naturalHeight || media.videoHeight}`;
            const duration = Number.isFinite(media.duration) ? `${media.duration.toFixed(1)} s` : "";
            info.textContent = [this.fileName(name), dimensions, duration].filter(Boolean).join(" · ");
        });
        media.addEventListener("error", () => { info.textContent = `${this.fileName(name)} · Browser preview unavailable. You can still save the file for processing.`; });
        this.detail.querySelector(".detail-preview").append(media);
        if (kind === "video") {
            const audio = name.replace("ref_video_", "ref_video_audio_");
            const panel = this.detail.querySelector(".detail-audio");
            const audioURL = this.mediaURL(audio);
            const title = document.createElement("b"); title.textContent = `Video soundtrack${audioURL ? " · " + this.mediaTag(audio) : ""}`; panel.append(title);
            if (audioURL) { const player = document.createElement("audio"); player.src = audioURL; player.controls = true; player.preload = "metadata"; panel.append(player); }
            if (this.imageActions[name] === "upload" && !this.imageActions[audio]) {
                const note = document.createElement("small"); note.textContent = "Extracted when saving if the video contains audio."; panel.append(note);
            }
            const change = document.createElement("button"); change.type = "button"; change.dataset.action = "chooseMedia"; change.dataset.id = audio;
            change.textContent = audioURL ? "Replace audio" : "Attach audio"; panel.append(change);
            if (audioURL) {
                const remove = document.createElement("button"); remove.type = "button"; remove.dataset.action = "removeImage"; remove.dataset.id = audio; remove.textContent = "Remove audio"; panel.append(remove);
            }
        }
        this.detail.showModal();
    }
    settingControl([name, title, type, min, max, step]) {
        const label = document.createElement("label"); label.textContent = title;
        const control = document.createElement(Array.isArray(type) ? "select" : "input"); control.dataset.setting = name;
        if (Array.isArray(type)) for (const value of type) { const option = document.createElement("option"); option.value = value; option.textContent = value; control.append(option); }
        else { control.type = type; if (type === "number") { control.min = min; control.max = max; control.step = step; } }
        control.onchange = () => {
            if (!control.reportValidity()) { this.settingsFromWidgets(); return; }
            this.value(name, control.type === "checkbox" ? control.checked : control.type === "number" ? Number(control.value) : control.value);
        };
        label.append(control); this.$(".settings").append(label);
    }
    settingsFromWidgets() {
        for (const control of this.root.querySelectorAll("[data-setting]")) {
            const value = this.value(control.dataset.setting);
            if (control.type === "checkbox") control.checked = !!value;
            else control.value = value;
        }
    }
    async guard(action) {
        if (!this.dirty) return action();
        this.pendingAction = action; this.$(".unsaved").hidden = false;
    }
    async save() {
        const form = new FormData(); form.set("collection", this.collection); form.set("prompt", this.$(".prompt").value);
        if (this.current) { form.set("entry", this.current.id); form.set("base", this.current.revision); }
        for (const [frame] of this.media) {
            form.set(`${frame}_action`, this.imageActions[frame] || "keep");
            if (this.files[frame]) form.set(frame, this.files[frame]);
        }
        const data = await (await request(this.referenceMode ? "/reference-entry" : "/entry", {method: "POST", body: form})).json();
        if (!this.disposed) { this.apply(data, true); this.message("Item saved in the ComfyUI session."); }
    }
    async action(action, id) {
        switch (action) {
            case "previewMedia": this.preview(id); break;
            case "removeDetail": { const name = this.detailName; this.closeDetail(); await this.action("removeImage", name); break; }
            case "save": await this.save(); break;
            case "new": await this.guard(() => { this.show(null); this.drafting = true; this.message("New draft. The queue uses the saved selection until you click Add."); }); break;
            case "select": await this.guard(async () => {
                const data = await (await request("/select", post({collection: this.collection, entry: id}))).json();
                if (!this.disposed) this.apply(data, true);
            }); break;
            case "delete": {
                const entry = id || this.current?.id; if (!entry) return;
                await this.guard(async () => {
                    const data = await (await request(`/entry/${entry}?${new URLSearchParams({collection: this.collection})}`, {method: "DELETE"})).json();
                    if (!this.disposed) this.apply(data, true);
                }); break;
            }
            case "removeImage": {
                const names = /^ref_video_\d+$/.test(id) ? [id, id.replace("ref_video_", "ref_video_audio_")] : [id];
                for (const name of names) {
                    if (this.urls[name]) { URL.revokeObjectURL(this.urls[name]); delete this.urls[name]; }
                    delete this.files[name];
                    if (name === id) this.imageActions[name] = "remove";
                    else delete this.imageActions[name];
                }
                const detailName = this.detailName;
                this.renderMedia(); this.markDirty();
                if (detailName && detailName !== id && this.hasMedia(detailName)) this.preview(detailName);
                else this.closeDetail();
                break;
            }
            case "refresh": await this.guard(() => this.refresh(true)); break;
            case "stay": this.pendingAction = null; this.$(".unsaved").hidden = true; break;
            case "saveContinue": case "discard": {
                if (action === "saveContinue") await this.save();
                const pending = this.pendingAction; this.pendingAction = null; this.$(".unsaved").hidden = true;
                this.dirty = false; await pending?.(); break;
            }
        }
    }
    dispose() {
        this.closeDetail(); this.picker.dispose(); this.clearMedia(this.root);
        this.disposed = true; clearInterval(this.timer); this.resizeObserver.disconnect(); this.releaseURLs();
        api.removeEventListener("executed", this.onExecuted); panels.delete(this);
    }
}

app.registerExtension({
    name: "YAFV.VideoPrompts",
    afterConfigureGraph() { for (const panel of panels) panel.activate(); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["YAFVVideoPrompts", "YAFVReferenceVideoPrompts"].includes(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args); this.videoPrompts = new PromptPanel(this, nodeData.name === "YAFVReferenceVideoPrompts"); return result;
        };
        for (const name of ["onAdded", "onConfigure"]) {
            const original = nodeType.prototype[name];
            nodeType.prototype[name] = function (...args) {
                const result = original?.apply(this, args);
                if (nodeData.name === "YAFVReferenceVideoPrompts") {
                    for (let i = this.outputs.length; i < nodeData.output.length; i++) this.addOutput(nodeData.output_name[i], nodeData.output[i]);
                }
                queueMicrotask(() => this.videoPrompts?.activate()); return result;
            };
        }
        const connected = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = connected?.apply(this, args); this.videoPrompts?.mode(); return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) { this.videoPrompts?.dispose(); return removed?.apply(this, args); };
    },
});
