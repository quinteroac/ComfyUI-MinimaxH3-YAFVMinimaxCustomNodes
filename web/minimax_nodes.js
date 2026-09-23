import { app } from "../../scripts/app.js";

const nodeTypes = new Set([
    "MiniMaxH3TwoPassSampler", "YAFVStoryboardPrompt", "YAFVMediaEditor",
    "YAFVVideoPrompts", "YAFVReferenceVideoPrompts", "YAFVH3VideoExtend",
    "YAFVH3EncodeAV", "MiniMaxH3VideoExtendPatched", "MiniMaxH3EncodeAVPatched",
    "YAFVProjectMediaTrim", "YAFVProjectContext", "YAFVProjectCommit",
    "YAFVProjectReview", "YAFVProjectMotionContext",
]);
const sheet = document.createElement("link");
sheet.rel = "stylesheet";
sheet.href = new URL("./minimax.css", import.meta.url).href;
document.head.append(sheet);

function applyTheme(node) {
    node.color = "#232329";
    node.bgcolor = "#18181c";
    for (const widget of node.widgets ?? []) {
        // Modern LiteGraph widgets expose instance-level palette getters.
        if ("background_color" in widget) {
            for (const [key, value] of Object.entries({
                background_color: "#232329", text_color: "#f4f4f5",
                secondary_text_color: "#b0b0bc",
            })) Object.defineProperty(widget, key, { configurable: true, get: () => value });
        }
        if (widget.element && !widget.type?.startsWith("yafv_")) {
            widget.element.classList.add("yafv-native-widget");
        }
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "YAFV.MiniMax.NodeTheme",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeTypes.has(nodeData.name)) return;
        nodeType.title_text_color = "#f4f4f5";
        Object.defineProperties(nodeType.prototype, {
            titleFontStyle: { configurable: true, get: () => '600 14px "DM Sans", Inter, Arial, sans-serif' },
            innerFontStyle: { configurable: true, get: () => '12px "DM Sans", Inter, Arial, sans-serif' },
        });
        for (const name of ["onNodeCreated", "onConfigure"]) {
            const original = nodeType.prototype[name];
            nodeType.prototype[name] = function (...args) {
                const result = original?.apply(this, args);
                applyTheme(this);
                return result;
            };
        }
    },
});
