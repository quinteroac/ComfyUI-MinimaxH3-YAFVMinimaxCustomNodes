import {app} from "/scripts/app.js";

app.registerExtension({
    name: "YAFV.ProjectContextCompatibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const inputs = {
            YAFVProjectContext: [],
            MiniMaxH3TwoPassSampler: [["context_latent_pass2", "LATENT"]],
            YAFVProjectCommit: [["latent_pass1", "LATENT"], ["latent_final", "LATENT"], ["trim_frames", "INT"]],
        }[nodeData.name];
        if (!inputs) return;
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configure?.apply(this, args);
            for (let i = this.outputs.length; i < nodeData.output.length; i++) {
                this.addOutput(nodeData.output_name[i], nodeData.output[i]);
            }
            for (const [name, type] of inputs) {
                if (!this.inputs.some(input => input.name === name)) this.addInput(name, type, {shape: 7});
            }
            const widget = this.widgets?.find(w => w.name === "scene_mode");
            const aliases = {"Continuar escena": "Continue scene", "Nueva escena": "New scene"};
            if (widget && aliases[widget.value]) widget.value = aliases[widget.value];
            return result;
        };
    },
});
