import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const source = await readFile(new URL("../web/minimax_nodes.js", import.meta.url), "utf8");
let extension;
vm.runInNewContext(source.replace(/^import .*;\n/, "").replaceAll("import.meta.url", '"http://localhost/extensions/yafv/minimax_nodes.js"'), {
    app: { registerExtension(value) { extension = value; } }, URL,
    document: { createElement: () => ({}), head: { append() {} } },
});
for (const name of [
    "MiniMaxH3TwoPassSampler", "YAFVStoryboardPrompt", "YAFVMediaEditor",
    "YAFVVideoPrompts", "YAFVReferenceVideoPrompts", "YAFVH3VideoExtend",
    "YAFVH3EncodeAV", "MiniMaxH3VideoExtendPatched", "MiniMaxH3EncodeAVPatched",
    "YAFVProjectMediaTrim", "YAFVProjectContext", "YAFVProjectCommit",
    "YAFVProjectReview", "YAFVProjectMotionContext",
]) {
    class Widget {
        get background_color() { return "green"; }
        value = 42;
    }
    class Node {
        widgets = [new Widget()];
        onNodeCreated() { this.created = true; return "created"; }
        onConfigure(data) { this.bgcolor = data.bgcolor; this.configured = true; return "configured"; }
        setDirtyCanvas() { this.dirty = true; }
    }
    extension.beforeRegisterNodeDef(Node, { name });
    const node = new Node();
    assert.equal(node.onNodeCreated(), "created", name);
    assert.equal(node.bgcolor, "#18181c", name);
    assert.equal(node.color, "#232329", name);
    assert.equal(node.widgets[0].background_color, "#232329", name);
    assert.equal(node.onConfigure({ bgcolor: "green" }), "configured", name);
    assert.equal(node.bgcolor, "#18181c", `${name}: saved colors migrate`);
    assert.equal(node.widgets[0].value, 42, `${name}: widget values preserved`);
    assert.ok(node.created && node.configured && node.dirty);
    assert.match(node.titleFontStyle, /DM Sans/);
}
class UnrelatedNode { onNodeCreated() {} }
const original = UnrelatedNode.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(UnrelatedNode, { name: "KSampler" });
assert.equal(UnrelatedNode.prototype.onNodeCreated, original);
assert.equal(UnrelatedNode.title_text_color, undefined);
console.log("Node theme checks passed: 14 node types, saved colors, lifecycle hooks, widget values and unrelated node isolation.");
