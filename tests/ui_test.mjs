// Browser-level checks without npm dependencies. Pass a Chromium binary as argv[2].
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import http from "node:http";
import { promisify } from "node:util";

const source = await readFile(new URL("../web/two_pass.js", import.meta.url), "utf8");
const harness = `
import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';
import '/extensions/yafv/two_pass.js';
try {
    const check = (condition, message) => { if (!condition) throw Error(message); };
    class Node {
        constructor() {
            this.id = 7; this.size = [400, 600]; this.inputs = [{name:'model_pass2',link:null}];
            this.widgets = Object.entries({
                enable_pass2:true,enable_latent_upscale:true,enable_pass1_preview:true,
                total_steps:8,split_step:4,seed_pass1:0,cfg_pass1:1,seed_pass2:0,cfg_pass2:1,
                upscale_mode:'scale by multiplier',upscale_scale:2,upscale_width:1280,
                upscale_height:704,upscale_megapixels:1,preview_fps:24,preview_audio:false,
                pass2_sampling_mode:'full',pass2_chunking_mode:'auto (chunk count)',pass2_chunk_count:2,
                pass2_chunk_frames:73,pass2_overlap_frames:22,
                pass2_audio_mode:'preserve',pass2_audio_steps:8,pass2_audio_start_step:4,
            }).map(([name,value])=>({name,value,type:'number',computeSize:()=>[200,20]}));
        }
        addDOMWidget(name,type,element,options) {
            document.body.append(element);
            const widget={name,type,options}; this.widgets.push(widget); return widget;
        }
        setSize(size){this.size=size;} computeSize(){return [400,600];} setDirtyCanvas(){}
    }
    await app.extension.beforeRegisterNodeDef(Node, {name:'MiniMaxH3TwoPassSampler'});
    const node = new Node(); node.onNodeCreated();
    const get = name=>node.widgets.find(w=>w.name===name);
    check(node.widgets[0].name==='enable_pass2','Widget serialization order changed');
    check(get('cfg_pass2').hidden,'Pass 2 fields should be hidden on Pass 1 tab');
    node.h3Panel.tabs.get('Upscale').click();
    check(!get('upscale_scale').hidden && get('upscale_width').hidden,'Multiplier controls');
    get('upscale_mode').value='target dimensions'; get('upscale_mode').callback();
    check(get('upscale_scale').hidden && !get('upscale_width').hidden,'Dimension controls');
    node.h3Panel.tabs.get('Pass 2').click();
    check(!get('pass2_sampling_mode').hidden && get('pass2_chunking_mode').hidden,'Full mode controls');
    get('pass2_sampling_mode').value='temporal'; get('pass2_sampling_mode').callback();
    check(!get('pass2_chunking_mode').hidden && !get('pass2_chunk_count').hidden && get('pass2_chunk_frames').hidden,'Automatic temporal controls');
    get('pass2_chunking_mode').value='manual (frames)'; get('pass2_chunking_mode').callback();
    check(!get('pass2_chunk_frames').hidden && !get('pass2_overlap_frames').hidden && get('pass2_chunk_count').hidden,'Manual temporal controls');
    check(node.h3Panel.summary.textContent.includes('audio preserved'),'Temporal audio status');
    check(!get('pass2_audio_mode').hidden && get('pass2_audio_steps').hidden,'Preserve audio controls');
    get('pass2_audio_mode').value='refine'; get('pass2_audio_mode').callback();
    check(!get('pass2_audio_steps').hidden && !get('pass2_audio_start_step').hidden,'Refine audio controls');
    check(node.h3Panel.summary.textContent.includes('audio refined'),'Refine audio status');
    api.dispatchEvent(new CustomEvent('yafv-h3-stage',{detail:{node:'7',run_id:'a',stage:'pass1',clear:true}}));
    api.dispatchEvent(new CustomEvent('yafv-h3-stage',{detail:{node:'7',run_id:'a',stage:'preview_ready',preview:{filename:'preview.mp4'}}}));
    check(!node.h3Panel.figure.hidden,'Live preview must appear');
    api.dispatchEvent(new CustomEvent('yafv-h3-stage',{detail:{node:'7',run_id:'a',stage:'audio_refine'}}));
    check(node.h3Panel.status.textContent.includes('Refining full audio'),'Audio refinement progress');
    api.dispatchEvent(new CustomEvent('yafv-h3-stage',{detail:{node:'7',run_id:'a',stage:'pass2'}}));
    check(!node.h3Panel.figure.hidden,'Pass 2 must retain Pass 1 preview');
    api.dispatchEvent(new CustomEvent('yafv-h3-stage',{detail:{node:'7',run_id:'a',stage:'pass2',chunk:2,chunks:7,frame_start:51,frame_end:124}}));
    check(node.h3Panel.status.textContent.includes('Chunk 2/7'),'Chunk progress');
    await node.h3Panel.stop.onclick(); check(api.interrupted,'Interrupt button');
    get('enable_pass1_preview').value=false; get('enable_pass1_preview').callback();
    check(node.h3Panel.figure.hidden && !node.h3Panel.video.hasAttribute('src'),'Disabled preview cleared');
    get('enable_pass2').value=false; get('enable_pass2').callback();
    check(get('split_step').hidden,'Disabled split step');
    check(get('pass2_chunk_frames').hidden && get('pass2_chunking_mode').hidden && get('pass2_sampling_mode').hidden,'Disabled Pass 2 settings');
    get('enable_pass1_preview').value=true; get('enable_pass1_preview').callback();
    node.onExecuted({h3_preview:[{filename:'cached.mp4'}]});
    check(!node.h3Panel.figure.hidden && node.h3Panel.stop.hidden,'Cached preview restored');
    node.onRemoved();
    document.body.dataset.result='passed';
} catch(error) { document.body.dataset.result='failed'; document.body.append(error.stack); }
`;
const routes = {
    "/": '<!doctype html><html><head></head><body style="width:400px;background:#0b1119;padding:20px"><script type="module" src="/test.js"></script></body></html>',
    "/scripts/app.js": "export const app={registerExtension(extension){this.extension=extension}};",
    "/scripts/api.js": "export const api=new EventTarget();api.apiURL=p=>p;api.interrupt=async()=>{api.interrupted=true};",
    "/extensions/yafv/two_pass.js": source,
    "/test.js": harness,
};
const server = http.createServer((request, response) => {
    const path = request.url.split("?")[0];
    response.writeHead(path in routes ? 200 : 404, {"Content-Type": path === "/" ? "text/html" : "text/javascript"});
    response.end(routes[path] ?? "");
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
try {
    const {stdout} = await promisify(execFile)(process.argv[2] ?? "chromium", [
        "--headless", "--no-sandbox", "--disable-gpu", "--dump-dom", "--virtual-time-budget=2000",
        `http://127.0.0.1:${server.address().port}/`,
    ], {timeout: 20000, maxBuffer: 1024 * 1024});
    assert.match(stdout, /data-result="passed"/, stdout);
    console.log("Browser UI checks passed: tabs, conditional fields, live/cached preview, interrupt, cleanup.");
} finally {
    server.closeAllConnections();
    server.close();
}
