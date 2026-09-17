import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "MiniMaxH3TwoPassSampler";
const panels = new Set();
const labels = {
    pass1: "Sampling · Pass 1", preview: "Decoding preview", preview_ready: "Pass 1 ready",
    upscale: "Upscaling latent", pass2: "Sampling · Pass 2", complete: "Complete",
    audio_refine: "Refining full audio · low-resolution video",
    cancelled: "Cancelled", error: "Execution failed",
};
const style = document.createElement("style");
style.textContent = `
.yafv-h3 { box-sizing:border-box; padding:14px; border:1px solid #374151; border-radius:14px;
    background:linear-gradient(145deg,#192432,#111821); color:#e8edf5; font:12px system-ui,sans-serif; }
.yafv-h3 * { box-sizing:border-box; }
.yafv-h3 header { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.yafv-h3 strong { font-size:14px; letter-spacing:.3px; }
.yafv-h3 .badge { color:#88e0cc; font-size:10px; letter-spacing:1px; }
.yafv-h3 .status { margin:10px 0 5px; color:#9cddcf; }
.yafv-h3 .summary { color:#aebccc; line-height:1.5; margin-bottom:12px; }
.yafv-h3 nav { display:flex; gap:5px; flex-wrap:wrap; }
.yafv-h3 button { cursor:pointer; padding:7px 10px; border:1px solid #425164;
    border-radius:8px; color:#cbd5e1; background:#202d3e; font:inherit; }
.yafv-h3 button[aria-selected="true"] { background:#285248; color:#c7fff0; border-color:#54a591; }
.yafv-h3 button:focus-visible { outline:2px solid #88e0cc; outline-offset:2px; }
.yafv-h3 button:disabled { cursor:default; opacity:.45; }
.yafv-h3 figure { margin:12px 0 0; }
.yafv-h3 video { width:100%; max-height:320px; display:block; border-radius:8px; background:#070b10; }
.yafv-h3 figcaption { margin:6px 0; color:#aebccc; }
.yafv-h3 .stop { margin-top:8px; border-color:#98515c; color:#ffd4da; background:#442a33; }
.yafv-h3 [hidden] { display:none !important; }
`;
document.head.append(style);

function value(node, name) {
    return node.widgets?.find(w => w.name === name)?.value;
}

function group(name) {
    if (name.startsWith("upscale_")) return "Upscale";
    if (name.startsWith("preview_")) return "Preview";
    if (name.includes("pass2")) return "Pass 2";
    return "Pass 1";
}

function refresh(panel) {
    const { node } = panel;
    const second = value(node, "enable_pass2");
    const upscale = value(node, "enable_latent_upscale");
    const preview = value(node, "enable_pass1_preview");
    const total = value(node, "total_steps");
    const split = value(node, "split_step");
    const temporal = value(node, "pass2_sampling_mode") === "temporal";
    const automaticChunks = value(node, "pass2_chunking_mode") === "auto (chunk count)";
    const refineAudio = value(node, "pass2_audio_mode") === "refine";
    panel.summary.textContent = `${second ? `${split} + ${total - split}` : total} steps · ` +
        `${upscale ? "upscale on" : "upscale off"} · ` +
        (second ? (node.inputs?.find(i => i.name === "model_pass2")?.link != null
            ? "separate Pass 2 model" : "shared model") : "Pass 2 off") +
        (second && temporal ? ` · temporal chunks · ${automaticChunks ? `${value(node, "pass2_chunk_count")} automatic` : `${value(node, "pass2_chunk_frames")} frames`} · audio ${refineAudio ? "refined" : "preserved"}` : "");
    if (!preview) panel.clearPreview();

    let seedGroup = "Pass 1";
    for (const widget of node.widgets ?? []) {
        if (widget === panel.widget) continue;
        if (!panel.originals.has(widget)) {
            panel.originals.set(widget, { type: widget.type, computeSize: widget.computeSize });
        }
        if (widget.name.startsWith("seed_pass")) seedGroup = group(widget.name);
        const belongs = widget.name === "control_after_generate" ? seedGroup : group(widget.name);
        let visible = widget.name.startsWith("enable_") || belongs === panel.tab;
        if (belongs === "Pass 2" && !widget.name.startsWith("enable_")) visible &&= !!second;
        if (belongs === "Upscale") visible &&= !!upscale;
        if (belongs === "Preview") visible &&= !!preview;
        if (["split_step", "pass1_return_with_leftover_noise"].includes(widget.name)) visible &&= !!second;
        if (widget.name === "pass2_chunking_mode") visible &&= temporal;
        if (widget.name === "pass2_chunk_count") visible &&= temporal && automaticChunks;
        if (["pass2_chunk_frames", "pass2_overlap_frames"].includes(widget.name)) visible &&= temporal && !automaticChunks;
        if (widget.name === "pass2_audio_mode") visible &&= temporal;
        if (["pass2_audio_steps", "pass2_audio_start_step"].includes(widget.name)) visible &&= temporal && refineAudio;
        const mode = value(node, "upscale_mode");
        if (widget.name === "upscale_scale") visible &&= mode === "scale by multiplier";
        if (["upscale_width", "upscale_height"].includes(widget.name)) visible &&= mode === "target dimensions";
        if (widget.name === "upscale_megapixels") visible &&= mode === "megapixels";
        const original = panel.originals.get(widget);
        widget.hidden = !visible;
        widget.type = visible ? original.type : "yafv_hidden";
        widget.computeSize = visible ? original.computeSize : () => [0, -4];
    }
    for (const [tab, button] of panel.tabs) button.setAttribute("aria-selected", String(tab === panel.tab));
    const rows = node.widgets.filter(w => w !== panel.widget && !w.hidden).length;
    const sockets = Math.max(node.inputs?.length ?? 0, node.outputs?.length ?? 0);
    node.setSize([Math.max(node.size[0], 380), Math.max(node.computeSize()[1], sockets * 20 + rows * 24 + panel.height() + 30)]);
    node.setDirtyCanvas(true, true);
}

function createPanel(node) {
    const root = document.createElement("section");
    root.className = "yafv-h3";
    root.innerHTML = `<header><strong>MiniMax H3</strong><span class="badge">TWO PASS</span></header>
        <div class="status" role="status">Ready</div><div class="summary"></div><nav aria-label="Sampler settings"></nav>
        <figure hidden><video controls loop playsinline preload="metadata"></video>
        <figcaption>Pass 1 · before upscale and refinement</figcaption></figure>
        <button type="button" class="stop" hidden>Stop current execution</button>`;
    const panel = {
        node, root, tab: "Pass 1", tabs: new Map(), originals: new Map(), active: false,
        summary: root.querySelector(".summary"), status: root.querySelector(".status"),
        video: root.querySelector("video"), figure: root.querySelector("figure"), stop: root.querySelector(".stop"),
        height() { return this.figure.hidden ? (this.active ? 186 : 148) : 520; },
        clearPreview() {
            this.video.pause();
            if (this.video.hasAttribute("src")) {
                this.video.removeAttribute("src");
                this.video.load();
            }
            this.figure.hidden = true;
        },
        showPreview(file) {
            if (!value(node, "enable_pass1_preview") || !file) return;
            const query = new URLSearchParams({ filename: file.filename, subfolder: file.subfolder ?? "", type: "temp" });
            this.video.src = api.apiURL(`/view?${query}`);
            this.figure.hidden = false;
        },
    };
    for (const tab of ["Pass 1", "Upscale", "Pass 2", "Preview"]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = tab;
        button.onclick = () => { panel.tab = tab; refresh(panel); };
        panel.tabs.set(tab, button);
        root.querySelector("nav").append(button);
    }
    panel.stop.title = "Interrupts the currently running ComfyUI prompt, including this node.";
    panel.stop.onclick = async () => {
        if (!panel.active) return;
        panel.stop.disabled = true;
        try {
            await api.interrupt();
            panel.status.textContent = "Cancellation requested…";
        } catch (error) {
            panel.status.textContent = `Could not interrupt: ${error.message}`;
            panel.stop.disabled = false;
        }
    };
    panel.widget = node.addDOMWidget("h3_panel", "yafv_panel", root, {
        serialize: false,
        getMinHeight: () => panel.height(),
        getHeight: () => panel.height(),
    });
    panel.widget.serialize = false;
    for (const widget of node.widgets) {
        if (widget === panel.widget) continue;
        const callback = widget.callback;
        widget.callback = function (...args) {
            const result = callback?.apply(this, args);
            refresh(panel);
            return result;
        };
    }
    panels.add(panel);
    refresh(panel);
    return panel;
}

api.addEventListener("yafv-h3-stage", ({ detail }) => {
    for (const panel of panels) {
        if (String(panel.node.id) !== detail.node) continue;
        if (detail.clear) {
            panel.runId = detail.run_id;
            panel.clearPreview();
        }
        if (panel.runId && panel.runId !== detail.run_id) continue;
        panel.active = !["complete", "cancelled", "error"].includes(detail.stage);
        panel.status.textContent = labels[detail.stage] ?? detail.stage;
        if (detail.stage === "pass2" && detail.chunk != null) {
            panel.status.textContent = `Pass 2 · Chunk ${detail.chunk}/${detail.chunks} · Frames ${detail.frame_start}–${detail.frame_end - 1}`;
        }
        panel.stop.hidden = !panel.active;
        panel.stop.disabled = false;
        if (detail.preview) panel.showPreview(detail.preview);
        refresh(panel);
    }
});

app.registerExtension({
    name: "YAFV.MiniMaxH3.TwoPass",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this.color = "#223a3a";
            this.bgcolor = "#142329";
            this.h3Panel = createPanel(this);
            return result;
        };
        for (const name of ["onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[name];
            nodeType.prototype[name] = function (...args) {
                const result = original?.apply(this, args);
                if (this.h3Panel) refresh(this.h3Panel);
                return result;
            };
        }
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.call(this, message);
            if (!this.h3Panel) return;
            this.h3Panel.active = false;
            this.h3Panel.stop.hidden = true;
            this.h3Panel.status.textContent = "Complete";
            if (message.h3_preview?.[0]) this.h3Panel.showPreview(message.h3_preview[0]);
            else this.h3Panel.clearPreview();
            refresh(this.h3Panel);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            this.h3Panel?.clearPreview();
            panels.delete(this.h3Panel);
            return removed?.apply(this, args);
        };
    },
});
