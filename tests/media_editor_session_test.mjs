// Exercise node lifecycle hooks without a ComfyUI server or browser dependencies.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../web/media_editor.js", import.meta.url), "utf8");
let extension;
const context = vm.createContext({
    app: { registerExtension(value) { extension = value; } },
    api: { removeEventListener() {} },
    document: { createElement: () => ({}), head: { append() {} } },
    URL, structuredClone, crypto, clearInterval, clearTimeout, cancelAnimationFrame() {},
});
vm.runInContext(source.replace(/^import .*;\n/gm, "")
    .replace("import.meta.url", JSON.stringify(new URL("../web/media_editor.js", import.meta.url).href))
    + "\nglobalThis.Editor = Editor;", context);

class Node {
    onSerialize(data) { data.originalSerialize = true; return "serialized"; }
    onConfigure() { this.originalConfigure = true; return "configured"; }
}
await extension.beforeRegisterNodeDef(Node, { name: "YAFVMediaEditor" });
function makeNode(graphId, nodeId = 7) {
    const node = new Node();
    node.graph = { id: graphId }; node.id = nodeId;
    const elements = new Map();
    const editor = Object.assign(Object.create(context.Editor.prototype), {
        node, sessionId: crypto.randomUUID(), clips: [], undo: [], redo: [],
        time: 0, selected: null, zoom: 60, fps: 24, width: 1280, height: 720,
        assets: new Map(), metadata: new Map(),
        $: key => { if (!elements.has(key)) elements.set(key, { value: "default" }); return elements.get(key); },
        gallery() {}, update() {}, timeline() {}, paint() {}, clearSource() {},
        resizeObserver: { disconnect() {} }, layoutObserver: { disconnect() {} },
    });
    node.mediaEditor = editor;
    return node;
}
function serialize(node) {
    const data = { id: node.id };
    assert.equal(node.onSerialize(data), "serialized");
    assert.equal(data.originalSerialize, true);
    assert.equal(data.clips, undefined, "Edits must stay in memory");
    return JSON.parse(JSON.stringify(data));
}
function configure(data, graphId) {
    const node = makeNode(graphId, data.id);
    assert.equal(node.onConfigure(data), "configured");
    assert.equal(node.originalConfigure, true);
    return node;
}

let first = makeNode("shared-graph");
const clip = { uid: "clip-a", in: 1, out: 4, strokes: [] };
Object.assign(first.mediaEditor, {
    clips: [clip], undo: [[{ ...clip, out: 5 }]], redo: [[{ ...clip, in: 2 }]],
    time: 1.5, selected: clip.uid, zoom: 100, fps: 30, width: 1920, height: 1080,
    strokeClip: clip, stroke: { points: [[0.1, 0.2]], color: "#ff0000" },
});
first.mediaEditor.$(".color").value = "#123456";
const firstData = serialize(first);
assert.equal(clip.strokes.length, 1, "An unfinished stroke is saved before serialization");
// ComfyUI may detach the node or mutate its graph before onRemoved.
first.graph = null;
first.onRemoved();
const second = makeNode("shared-graph");
second.mediaEditor.clips = [{ uid: "clip-b" }];
const secondData = serialize(second);
second.onRemoved();

for (let i = 0; i < 3; i++) {
    first = configure(firstData, `recreated-graph-${i}`);
    const editor = first.mediaEditor;
    assert.equal(editor.clips[0].uid, "clip-a");
    assert.equal(editor.clips[0].strokes[0].color, "#ff0000");
    assert.equal(editor.undo[0][0].out, 5);
    assert.equal(editor.redo[0][0].in, 2);
    for (const [key, value] of Object.entries({ time: 1.5, selected: "clip-a", zoom: 100, fps: 30, width: 1920, height: 1080 })) {
        assert.equal(editor[key], value);
    }
    assert.equal(editor.$(".color").value, "#123456");
    serialize(first); first.onRemoved();
    const other = configure(secondData, "shared-graph");
    assert.equal(other.mediaEditor.clips[0].uid, "clip-b", "Same node/graph IDs must not mix editors");
    other.onRemoved();
}
const fresh = configure({ id: 7 }, "shared-graph");
assert.equal(fresh.mediaEditor.clips.length, 0, "A workflow without a session starts empty");
const copy = configure({ ...firstData, id: 8 }, "shared-graph");
assert.equal(copy.mediaEditor.clips.length, 0, "A duplicated node has an independent session");
console.log("Editor session checks passed: repeated tab reconstruction, isolation, strokes, undo/redo, settings, detached disposal and existing hooks.");
