import {app} from "/scripts/app.js";
import {api} from "/scripts/api.js";

const sheet = document.createElement("link");
sheet.rel = "stylesheet";
sheet.href = new URL("./minimax.css", import.meta.url).href;
document.head.append(sheet);

const mounted = new Set();
let recovering = false;
function deliver(data) {
    const node = findNode(data.node_id) || findMountedNode(data.node_id);
    node?._yafvProjectReview?.(data);
}
async function recover() {
    if (recovering || !mounted.size) return;
    recovering = true;
    try {
        const response = await api.fetchApi("/yafv/project-review/pending");
        if (response.ok) for (const data of await response.json()) deliver(data);
    } catch (_) { /* Retry on the next poll or reconnect. */ }
    finally { recovering = false; }
}

function findNode(id) {
    return app.graph?.getNodeById?.(Number(String(id).split(":").at(-1)));
}

function findMountedNode(id) {
    const target = String(id).split(":").at(-1);
    return [...mounted].find(node => String(node.id) === target || String(node._yafvReviewId) === String(id));
}

app.registerExtension({
    name: "YAFV.ProjectReview",
    setup() {
        api.addEventListener("yafv_project_review", event => {
            const data = event.detail || {};
            deliver(data);
        });
        api.addEventListener("yafv_project_review_closed", event => {
            for (const node of mounted) node._yafvProjectReviewClose?.(event.detail.token);
        });
        api.addEventListener("reconnected", recover);
        setInterval(recover, 3000);
        api.addEventListener("execution_interrupted", () => {
            for (const node of mounted) node._yafvProjectReviewCancel?.();
        });
        api.addEventListener("execution_error", () => {
            for (const node of mounted) node._yafvProjectReviewCancel?.();
        });
    },
    async nodeCreated(node) {
        if (node.comfyClass !== "YAFVProjectReview" || mounted.has(node)) return;
        mounted.add(node);
        const root = document.createElement("section");
        root.className = "yafv-review";
        root.innerHTML = `<header><strong>MiniMax · Project review</strong></header><video controls preload="metadata" style="width:100%;max-height:220px;background:#080b10;object-fit:contain"></video><div style="display:flex;gap:8px"><button data-action="approve">Approve</button><button data-action="reject">Reject</button></div><span role="status">Waiting for a generated candidate…</span>`;
        const video = root.querySelector("video");
        const status = root.querySelector("[role=status]");
        const buttons = [...root.querySelectorAll("button[data-action]")];
        const stop = document.createElement("button");
        stop.textContent = "Stop workflow";
        stop.title = "Cancel the running job, even without a video or pending review";
        root.insertBefore(stop, status);
        stop.onclick = async () => {
            status.textContent = "Requesting cancellation…";
            try {
                const response = await api.fetchApi("/yafv/project-review/stop", {method: "POST"});
                if (!response.ok) throw new Error("Could not stop the workflow");
                current = null;
                setEnabled(false);
                status.textContent = "Cancellation requested.";
            } catch (error) { status.textContent = error.message; }
        };
        let current = null;
        const setEnabled = enabled => buttons.forEach(button => { button.disabled = !enabled; });
        setEnabled(false);
        for (const button of buttons) button.onclick = async () => {
            if (!current) return;
            const token = current;
            setEnabled(false);
            try {
                const response = await api.fetchApi("/yafv/project-review", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({token, action: button.dataset.action}),
                });
                if (!response.ok) throw new Error("Review request failed");
                status.textContent = button.dataset.action === "approve" ? "Approved; committing segment…" : "Rejected; segment was not committed.";
            } catch (error) {
                status.textContent = error.message;
                if (current === token) setEnabled(true);
            }
        };
        root.addEventListener("pointerdown", event => event.stopPropagation());
        node.addDOMWidget("project_review_panel", "yafv_project_review", root, {serialize: false, getMinHeight: () => 400});
        node._yafvProjectReview = data => {
            if (current === data.token) return;
            current = data.token;
            video.src = data.video_preview;
            video.load();
            status.textContent = "Review the candidate and choose an action.";
            setEnabled(true);
        };
        node._yafvReviewId = node.id;
        node._yafvProjectReviewClose = token => {
            if (current !== token) return;
            current = null;
            setEnabled(false);
            status.textContent = "Review finished.";
        };
        node._yafvProjectReviewCancel = () => {
            const token = current;
            current = null;
            setEnabled(false);
            if (token) {
                void api.fetchApi("/yafv/project-review/cancel", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({token}),
                }).catch(() => {});
            }
        };
        node.setSize?.([360, 360]);
        const removed = node.onRemoved;
        node.onRemoved = function (...args) {
            mounted.delete(node);
            return removed?.apply(this, args);
        };
        void recover();
    },
});
