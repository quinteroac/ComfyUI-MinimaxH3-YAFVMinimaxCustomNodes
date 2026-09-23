import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

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
          <section class="form">${referenceMode ? `<div class="reference-toolbar"><button data-action="toggleReferences" aria-expanded="false">+ Add reference</button><span class="reference-count muted"></span><div class="reference-menu" hidden><button data-action="addReference" data-id="image">Image</button><button data-action="addReference" data-id="video">Video</button><button data-action="addReference" data-id="audio">Audio</button></div><input class="reference-picker" type="file" hidden></div>` : ""}<div class="frames"></div>
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
            if (button) this.run(() => this.action(button.dataset.action, button.dataset.id));
        });
        this.$(".prompt").oninput = () => this.markDirty();
        if (referenceMode) {
            this.$(".reference-picker").onchange = event => {
                const input = event.target, kind = input.dataset.kind;
                const slot = this.availableSlot(kind);
                if (slot && input.files[0]) this.setFile(slot[0], kind, input.files[0]);
                input.value = "";
            };
        } else {
            for (const [name, kind, label] of this.media) this.frameControl(name, kind, label);
        }
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
        for (const button of this.root.querySelectorAll("button")) button.disabled = this.busy || !this.hydrated;
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
                this.$(`[data-action="addReference"][data-id="${kind}"]`).disabled ||= count === slots.length;
                return `${count}/${slots.length} ${{image: "images", video: "videos", audio: "audios"}[kind]}`;
            });
            this.$(".reference-count").textContent = counts.join(" · ");
        }
        this.mode();
    }
    mode() {
        const connected = name => this.node.inputs?.find(input => input.name === name)?.link != null;
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
                ? this.media.filter(([name]) => entry[name]).map(([name]) => name).join(" · ") || "No references"
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
        if (this.referenceMode) {
            this.clearMedia(this.$(".frames"));
            this.$(".frames").replaceChildren();
            this.$(".reference-menu").hidden = true;
            this.$('[data-action="toggleReferences"]').setAttribute("aria-expanded", "false");
            this.renderMedia();
        } else {
            for (const [frame] of this.media) this.preview(frame);
        }
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
        for (const [name, kind, label] of this.media) {
            const soundtrack = name.startsWith("ref_video_audio_");
            const video = soundtrack ? name.replace("ref_video_audio_", "ref_video_") : null;
            const parent = soundtrack && this.hasMedia(video) ? this.$(`.${video} .soundtrack-container`) : this.$(".frames");
            const visible = this.hasMedia(name) || (soundtrack && this.hasMedia(video));
            let section = this.$(`.${name}`);
            if (section && (!visible || section.parentElement !== parent)) {
                this.clearMedia(section); section.remove(); section = null;
            }
            if (visible && !section) {
                this.frameControl(name, kind, label, parent, soundtrack && this.hasMedia(video));
                this.preview(name);
            }
        }
    }
    setFile(frame, kind, file) {
        if (!file || this.busy || !this.hydrated || this.disposed) return;
        if (file.type && !file.type.startsWith(`${kind}/`) && file.type !== "application/octet-stream") return this.message(`Select a ${kind} file.`, true);
        if (this.urls[frame]) URL.revokeObjectURL(this.urls[frame]);
        this.files[frame] = file; this.imageActions[frame] = "upload"; this.urls[frame] = URL.createObjectURL(file);
        if (this.referenceMode) this.renderMedia();
        this.preview(frame);
        if (kind === "video") this.preview(frame.replace("ref_video_", "ref_video_audio_"));
        this.markDirty();
    }
    preview(frame) {
        const section = this.$(`.${frame}`);
        if (!section) return;
        const element = section.querySelector(".media-preview");
        const video = frame.startsWith("ref_video_audio_") ? frame.replace("ref_video_audio_", "ref_video_") : null;
        const autoAudio = video && this.imageActions[video] === "upload" && !this.imageActions[frame];
        const url = this.urls[frame] || (autoAudio ? this.urls[video] : this.hasMedia(frame) && this.current?.[frame] ? api.apiURL(`${BASE}/${this.referenceMode ? "media" : "image"}/${this.current.revision}/${frame}?${new URLSearchParams({collection: this.collection})}`) : null);
        element.hidden = !url;
        if (element.tagName !== "IMG") element.pause();
        if (url) element.src = url; else element.removeAttribute("src");
        if (element.tagName !== "IMG") element.load();
        section.querySelector(".media-note").textContent = autoAudio ? "Extracted when saving if the video contains audio." : "";
        if (section.classList.contains("soundtrack")) {
            section.querySelector('[data-action="removeImage"]').hidden = !this.hasMedia(frame);
            section.querySelector(".choose-file").textContent = this.hasMedia(frame) ? "Replace audio" : "Attach audio";
        }
    }
    frameControl(frame, kind, label, parent = this.$(".frames"), embedded = false) {
        const section = document.createElement("div"); section.className = `${embedded ? "soundtrack" : "frame"} ${frame}`;
        const tag = kind === "image" ? "img" : kind;
        section.innerHTML = `<b>${label}</b>${embedded ? `<button class="choose-file">Attach audio</button>` : `<div class="drop-zone" role="button" tabindex="0" aria-label="Load ${frame}"><span>Drag a file<br>or click to ${this.referenceMode ? "replace" : "load"}</span>${kind === "image" ? `<img class="media-preview" hidden alt="${frame}">` : ""}</div>`}${kind !== "image" ? `<${tag} class="media-preview" controls preload="metadata" hidden></${tag}>` : ""}<small class="media-note"></small><input type="file" accept="${kind}/*" hidden><button data-action="removeImage" data-id="${frame}">Remove ${kind === "image" ? "image" : kind}</button>${kind === "video" ? '<div class="soundtrack-container"></div>' : ""}`;
        const input = section.querySelector("input"), drop = section.querySelector(".drop-zone, .choose-file");
        drop.onclick = () => { if (!this.busy && this.hydrated) input.click(); };
        if (!embedded) drop.onkeydown = event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); drop.click(); } };
        input.onchange = () => { this.setFile(frame, kind, input.files[0]); input.value = ""; };
        drop.ondragover = event => { event.preventDefault(); event.stopPropagation(); drop.classList.add("drag"); };
        drop.ondragleave = () => drop.classList.remove("drag");
        drop.ondrop = event => { event.preventDefault(); event.stopPropagation(); drop.classList.remove("drag"); this.setFile(frame, kind, event.dataTransfer.files[0]); };
        parent.append(section);
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
            case "toggleReferences": {
                const menu = this.$(".reference-menu"); menu.hidden = !menu.hidden;
                this.$('[data-action="toggleReferences"]').setAttribute("aria-expanded", String(!menu.hidden));
                break;
            }
            case "addReference": {
                if (!this.availableSlot(id)) break;
                this.$(".reference-menu").hidden = true;
                this.$('[data-action="toggleReferences"]').setAttribute("aria-expanded", "false");
                const input = this.$(".reference-picker"); input.dataset.kind = id; input.accept = `${id}/*`; input.click();
                break;
            }
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
                if (this.referenceMode) this.renderMedia();
                this.preview(id); this.markDirty(); break;
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
        this.clearMedia(this.root);
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
