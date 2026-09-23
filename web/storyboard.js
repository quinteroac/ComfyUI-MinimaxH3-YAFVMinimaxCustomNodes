import { app } from "../../scripts/app.js";

const TYPE = "YAFVStoryboardPrompt";
const panels = new Set();
const numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight"];
const css = document.createElement("link");
css.rel = "stylesheet";
css.href = new URL("./storyboard.css", import.meta.url).href;
document.head.append(css);

const get = (node, name) => node.widgets?.find(item => item.name === name);
function put(node, name, value) {
    const item = get(node, name);
    if (item) item.value = value;
    node.graph?.setDirtyCanvas(true, true);
}
function makePrompt(panel) {
    const count = Math.max(1, Math.min(8, Number(get(panel.node, "scene_count")?.value ?? 4)));
    const description = String(get(panel.node, "general_scene_description")?.value ?? "").trim() || "[general scene description]";
    const style = String(get(panel.node, "style")?.value ?? "").trim() || "[style]";
    const lines = [`A ${numbers[count - 1]}-scene of ${description}`, ""];
    for (let i = 1; i <= count; i++) {
        const text = String(get(panel.node, `scene_${i}`)?.value ?? "").trim();
        lines.push(`Scene ${i}: ${text || `[Scene${i} user prompt]`}`, "");
    }
    lines.push(`Clean storyboard panel borders, professional ${style} storyboard sheet, cinematic framing notes, coherent visual storytelling, highly detailed ${style} aesthetic, consistent characters across all six scenes.`);
    return lines.join("\n");
}
function refresh(panel, reset = false) {
    const count = Math.max(1, Math.min(8, Number(get(panel.node, "scene_count")?.value ?? 4)));
    panel.count.value = count;
    panel.scenes.replaceChildren();
    for (let i = 1; i <= count; i++) {
        const card = document.createElement("label");
        card.className = "scene-card";
        card.innerHTML = `<span><b>Scene ${i}</b><small>Panel ${String(i).padStart(2, "0")}</small></span><textarea placeholder="Describe action, framing, characters and continuity…"></textarea>`;
        const input = card.querySelector("textarea");
        input.value = get(panel.node, `scene_${i}`)?.value ?? "";
        input.oninput = () => { put(panel.node, `scene_${i}`, input.value); if (!panel.dirty) update(panel); };
        panel.scenes.append(card);
    }
    if (reset || !panel.dirty) update(panel);
    panel.dirty = false;
    panel.status.textContent = `${count} ${count === 1 ? "scene" : "scenes"} · ready to edit`;
    panel.node.setSize([Math.max(panel.node.size[0], 440), Math.max(panel.node.computeSize()[1], 690)]);
    panel.node.setDirtyCanvas(true, true);
}
function update(panel) {
    panel.generated = makePrompt(panel);
    if (!panel.dirty) panel.preview.value = panel.generated;
}
function createPanel(node) {
    const root = document.createElement("section");
    root.className = "yafv-storyboard";
    root.innerHTML = `<header><div><span class="eyebrow">YAFV · PROMPT BUILDER</span><strong>Storyboard studio</strong></div><span class="badge">TEXT ONLY</span></header><div class="intro">Build a consistent storyboard prompt panel by panel. The final text stays fully editable.</div><div class="toolbar"><label>Scenes <select class="count">${Array.from({length: 8}, (_, i) => `<option value="${i + 1}">${i + 1}</option>`).join("")}</select></label><span class="status" role="status"></span><button class="secondary reset" type="button">Reset edit</button><button class="primary regenerate" type="button">Regenerate structure</button></div><label class="field"><span>General scene description</span><textarea class="description" placeholder="A team of explorers discovers a hidden city in the jungle…"></textarea></label><label class="field"><span>Visual style</span><input class="style" placeholder="cinematic, hand-painted concept art"></label><div class="section-title"><span>Scene panels</span><small>Describe each moment; characters remain coherent across the sheet.</small></div><div class="scene-list"></div><div class="section-title output-title"><span>Editable output</span><small>This exact text is sent through the output socket.</small></div><textarea class="preview" aria-label="Editable storyboard prompt"></textarea>`;
    const panel = {node, root, count: root.querySelector(".count"), scenes: root.querySelector(".scene-list"), preview: root.querySelector(".preview"), status: root.querySelector(".status"), dirty: false, generated: ""};
    root.addEventListener("pointerdown", event => event.stopPropagation());
    root.addEventListener("wheel", event => event.stopPropagation());
    root.addEventListener("keydown", event => event.stopPropagation());
    const description = root.querySelector(".description");
    const style = root.querySelector(".style");
    description.value = get(node, "general_scene_description")?.value ?? "";
    style.value = get(node, "style")?.value ?? "";
    description.oninput = () => { put(node, "general_scene_description", description.value); if (!panel.dirty) update(panel); };
    style.oninput = () => { put(node, "style", style.value); if (!panel.dirty) update(panel); };
    panel.count.onchange = () => { put(node, "scene_count", Number(panel.count.value)); refresh(panel, true); };
    panel.preview.oninput = () => { panel.dirty = true; put(node, "edited_prompt", panel.preview.value); panel.status.textContent = "Manual edit · saved to output"; };
    root.querySelector(".regenerate").onclick = () => { panel.dirty = false; put(node, "edited_prompt", ""); refresh(panel, true); };
    root.querySelector(".reset").onclick = () => { panel.dirty = false; put(node, "edited_prompt", ""); update(panel); panel.status.textContent = "Generated structure restored"; };
    panel.widget = node.addDOMWidget("storyboard_panel", "yafv_storyboard", root, {serialize: false, getMinHeight: () => 650});
    panel.widget.serialize = false;
    for (const item of node.widgets ?? []) if (item !== panel.widget) { item.hidden = true; item.type = "yafv_hidden"; item.computeSize = () => [0, -4]; }
    panels.add(panel);
    refresh(panel, true);
    return panel;
}

app.registerExtension({
    name: "YAFV.StoryboardPrompt",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TYPE) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) { const result = created?.apply(this, args); this.color = "#3b3150"; this.bgcolor = "#201a2d"; this.storyboardPanel = createPanel(this); return result; };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) { const result = configured?.apply(this, args); if (this.storyboardPanel) refresh(this.storyboardPanel, true); return result; };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) { panels.delete(this.storyboardPanel); return removed?.apply(this, args); };
    },
});
