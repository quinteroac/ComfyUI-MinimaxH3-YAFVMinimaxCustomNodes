import {app} from "/scripts/app.js";

app.registerExtension({
    name: "YAFV.ProjectContextCompatibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "YAFVProjectContext") return;
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configure?.apply(this, args);
            const widget = this.widgets?.find(w => w.name === "scene_mode");
            const aliases = {"Continuar escena": "Continue scene", "Nueva escena": "New scene"};
            if (widget && aliases[widget.value]) widget.value = aliases[widget.value];
            return result;
        };
    },
});
