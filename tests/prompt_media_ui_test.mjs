// Browser regression checks for prompt media selection and lifecycle.
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import http from "node:http";
import { promisify } from "node:util";

const files = Object.fromEntries(await Promise.all(["video_prompts.js", "prompt_media_picker.js", "video_prompts.css"].map(async name => [`/extensions/yafv/${name}`, await readFile(new URL(`../web/${name}`, import.meta.url), "utf8")])));
const harness = `
import { app } from '/scripts/app.js';
import '/extensions/yafv/video_prompts.js';
const tick = () => new Promise(resolve => setTimeout(resolve, 30));
const check = (condition, message) => { if (!condition) throw Error(message); };
class BaseNode {
    constructor(id) {
        this.id=id; this.graph={id:'test'}; this.inputs=[{name:'context_image',link:null}]; this.outputs=[];
        this.widgets=['collection_id','revision_id'].map(name=>({name,value:''}));
    }
    addDOMWidget(name,type,element) { document.body.append(element); element.style.width='1000px'; element.style.height='840px'; return {name,type}; }
    setSize() {} setDirtyCanvas() {}
}
try {
    class ReferenceNode extends BaseNode {}
    await app.extension.beforeRegisterNodeDef(ReferenceNode,{name:'YAFVReferenceVideoPrompts'});
    const node=new ReferenceNode(1); node.onNodeCreated(); await tick();
    const panel=node.videoPrompts;
    check(panel.hydrated,'Panel hydrated');
    check(panel.$('.media-add') && !panel.$('.frame'),'Empty gallery');
    const image=new File([await (await fetch('/pixel.png')).blob()],'scene.png',{type:'image/png'});
    panel.addDroppedFiles([image,image]);
    check(panel.$('.frames').querySelectorAll('.frame').length===2,'Multiple dropped images');
    check(panel.$('.ref_image_0').textContent.includes('<Picture 1>'),'Real MiniMax reference tag');
    check(!panel.$('.frame').draggable,'No free reordering');
    panel.$('[data-action="previewMedia"]').click(); await tick();
    check(panel.detail.open && panel.detail.querySelector('img'),'Image detail');
    panel.detail.querySelector('.detail-close').click();
    check(!panel.detail.open,'Close detail');
    panel.$('[data-action="chooseMedia"]').click(); await tick();
    check(panel.picker.dialog.open,'Replace opens picker');
    check(!panel.picker.$('[data-source="input"]').disabled,'Picker remains interactive while panel is busy');
    panel.picker.$('[data-source="input"]').click(); await tick();
    check(panel.picker.$('.browser-file'),'Inputs listed');
    panel.picker.$('[data-file-key]').click(); panel.picker.$('.picker-apply').click(); await tick(); await tick();
    check(!panel.picker.dialog.open && panel.files.ref_image_0.name==='input.png','Input selected and loaded');
    panel.$('[data-action="addReference"][data-id="image"]').click(); await tick();
    panel.picker.$('[data-source="output"]').click(); await tick();
    panel.picker.$('[data-file-key]').click(); panel.picker.$('.picker-apply').click(); await tick(); await tick();
    check(panel.files.ref_image_2.name==='output.png','Output selected in next stable slot');
    await panel.run(()=>panel.action('removeImage','ref_image_0'));
    check(!panel.hasMedia('ref_image_0') && panel.hasMedia('ref_image_1'),'Remove preserves other slots');
    check(panel.$('.ref_image_1').textContent.includes('<Picture 1>'),'Tags match backend after deletion');
    const video=new File(['video'],'clip.mp4',{type:'video/mp4'});
    panel.setFile('ref_video_0','video',video);
    panel.setFile('ref_video_audio_0','audio',new File(['audio'],'voice.wav',{type:'audio/wav'}));
    panel.preview('ref_video_0');
    check(panel.detail.querySelector('video') && panel.detail.querySelector('audio'),'Video detail includes attached audio');
    panel.closeDetail();
    panel.setFile('ref_video_0','video',video);
    check(!panel.files.ref_video_audio_0 && !panel.imageActions.ref_video_audio_0,'Replacing video resets soundtrack override');
    await panel.run(()=>panel.action('removeImage','ref_video_audio_0'));
    check(!panel.hasMedia('ref_video_audio_0') && panel.hasMedia('ref_video_0'),'Remove audio preserves video');
    panel.$('.prompt').value='Scene';
    await panel.run(()=>panel.action('save'));
    check(panel.current.media_names.ref_image_1==='scene.png','Saved media names retained');
    panel.preview('ref_image_1');
    check(panel.detail.querySelector('.detail-info').textContent==='scene.png','Saved preview shows filename');
    panel.closeDetail();
    class VideoNode extends BaseNode {}
    await app.extension.beforeRegisterNodeDef(VideoNode,{name:'YAFVVideoPrompts'});
    const videoNode=new VideoNode(2); videoNode.onNodeCreated(); await tick();
    const frames=videoNode.videoPrompts;
    check(frames.$('.frames').children.length===2,'Fixed start/end slots');
    frames.setFile('first','image',image); frames.setFile('last','image',image);
    videoNode.inputs[0].link=10; videoNode.onConnectionsChange();
    check(!frames.$('.context-note').hidden && frames.$('.first').classList.contains('context-override'),'Context override displayed');
    await frames.run(()=>frames.action('removeImage','first'));
    check(frames.hasMedia('last') && frames.$('.frames').children.length===2,'Removing start does not shift end');
    frames.$('[data-action="chooseMedia"]').click(); await tick();
    frames.picker.$('.picker-cancel').click(); await tick();
    check(!frames.busy,'Cancel unlocks panel');
    node.onRemoved(); videoNode.onRemoved();
    check(!panel.picker.dialog.isConnected && !panel.detail.open && !Object.keys(panel.urls).length,'Dispose cleans media and dialogs');
    document.body.dataset.result='passed';
} catch(error) { document.body.dataset.result='failed'; document.body.append(error.stack); }
`;
const routes = {
    ...files,
    "/": '<!doctype html><html><body style="background:#111a25"><script type="module" src="/test.js"></script></body></html>',
    "/scripts/app.js": "export const app={registerExtension(extension){this.extension=extension}};",
    "/scripts/api.js": `export const api=new EventTarget(); api.apiURL=p=>p; let entries=[]; api.fetchApi=async(p,o)=>{
      if (p.includes('/library')) return Response.json({epoch:'test',entries:[],selected:null});
      if (p.endsWith('/reference-entry')||p.endsWith('/entry')) {
        const f=o.body; const entry={id:'a',revision:'r',prompt:f.get('prompt'),media_names:{}};
        for (const [key,value] of f) if (value instanceof File) { entry[key]=true; entry.media_names[key]=value.name; }
        return Response.json({epoch:'test',entries:[entry],selected:'a'});
      }
      return fetch(p,o);
    };`,
    "/test.js": harness,
};
const pixel = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII=', 'base64');
const server = http.createServer((request, response) => {
    const url = new URL(request.url, 'http://localhost');
    if (url.pathname === '/yafv/prompts/browse') {
        response.setHeader('Content-Type','application/json');
        response.end(JSON.stringify({total:1,entries:[{name:url.searchParams.get('source')+'.png',path:'pixel.png',size:pixel.length}]})); return;
    }
    if (url.pathname === '/pixel.png' || url.pathname.includes('/browse-file') || url.pathname.includes('/media/')) {
        response.setHeader('Content-Type','image/png'); response.end(pixel); return;
    }
    const path=url.pathname;
    response.writeHead(path in routes ? 200 : 404, {'Content-Type':path==='/'?'text/html':path.endsWith('.css')?'text/css':'text/javascript'});
    response.end(routes[path] ?? '');
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
try {
    const {stdout}=await promisify(execFile)(process.argv[2] ?? 'chromium', [
        '--headless','--no-sandbox','--disable-gpu','--dump-dom','--virtual-time-budget=6000',
        ...(process.argv[3] ? [`--screenshot=${process.argv[3]}`] : []),
        '--window-size=1200,1000',`http://127.0.0.1:${server.address().port}/`,
    ], {timeout:25000,maxBuffer:2*1024*1024});
    assert.match(stdout,/data-result="passed"/,stdout);
    console.log('Prompt media UI checks passed: galleries, picker inputs/outputs, tags, audio, save, context, cleanup.');
} finally { server.closeAllConnections(); server.close(); }
