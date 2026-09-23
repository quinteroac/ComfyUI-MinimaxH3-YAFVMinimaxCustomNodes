import { api } from "../../scripts/api.js";

const BASE = "/yafv/prompts";

// One picker per panel; selections are applied only after the user clicks Add.
export class PromptMediaPicker {
    constructor(root) {
        this.dialog = document.createElement("dialog");
        this.dialog.className = "media-picker";
        this.dialog.innerHTML = `<div class="dialog-heading"><strong>Select media</strong><button type="button" class="picker-close" aria-label="Close media selector">×</button></div>
          <div class="picker-tabs" role="tablist" aria-label="Media source"><button type="button" role="tab" data-source="upload">Upload</button><button type="button" role="tab" data-source="input">Inputs</button><button type="button" role="tab" data-source="output">Outputs</button></div>
          <div class="picker-upload"><button type="button" class="upload-target">Drop files here or browse your device</button><input type="file" hidden></div>
          <div class="picker-browser" hidden><div class="browser-tools"><button type="button" class="folder-up" aria-label="Parent folder">↑</button><span class="browser-path"></span><input type="search" placeholder="Search this folder…" aria-label="Search this folder"><button type="button" class="browser-refresh" aria-label="Refresh files">↻</button></div><div class="browser-grid"></div><div class="browser-pages"><button type="button" class="page-prev">Previous</button><span></span><button type="button" class="page-next">Next</button></div></div>
          <p class="picker-status" role="status"></p><div class="dialog-actions"><span class="picker-selection"></span><button type="button" class="picker-cancel">Cancel</button><button type="button" class="picker-apply primary">Add selected</button></div>`;
        root.append(this.dialog);
        this.$ = selector => this.dialog.querySelector(selector);
        this.videoObserver = new IntersectionObserver(entries => {
            for (const {target, isIntersecting} of entries) if (isIntersecting) {
                target.src = target.dataset.src; this.videoObserver.unobserve(target);
            }
        }, {root: this.$(".browser-grid")});
        this.selected = new Map(); this.source = "upload"; this.path = ""; this.page = 0;
        this.$(".picker-close").onclick = this.$(".picker-cancel").onclick = () => this.close(null);
        this.dialog.addEventListener("cancel", event => { event.preventDefault(); this.close(null); });
        this.dialog.addEventListener("click", event => { if (event.target === this.dialog) this.close(null); });
        const tabs = [...this.dialog.querySelectorAll("[data-source]")];
        for (const [index, tab] of tabs.entries()) {
            tab.onclick = () => this.tab(tab.dataset.source);
            tab.onkeydown = event => {
                const next = {ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1}[event.key];
                if (next !== undefined) { event.preventDefault(); tabs[next].focus(); tabs[next].click(); }
            };
        }
        this.$(".upload-target").onclick = () => this.$('input[type="file"]').click();
        this.$('input[type="file"]').onchange = event => { this.upload(event.target.files); event.target.value = ""; };
        this.$(".picker-upload").ondragover = event => { event.preventDefault(); event.stopPropagation(); };
        this.$(".picker-upload").ondrop = event => { event.preventDefault(); event.stopPropagation(); this.upload(event.dataTransfer.files); };
        this.$(".folder-up").onclick = () => { this.path = this.path.split("/").slice(0, -1).join("/"); this.page = 0; this.load(); };
        this.$(".browser-refresh").onclick = () => this.load();
        this.$('input[type="search"]').oninput = () => { clearTimeout(this.searchTimer); this.searchTimer = setTimeout(() => { this.page = 0; this.load(); }, 200); };
        this.$(".page-prev").onclick = () => { this.page--; this.load(); };
        this.$(".page-next").onclick = () => { this.page++; this.load(); };
        this.$(".picker-apply").onclick = () => this.apply();
    }
    open(kind, count, replacing = false) {
        this.kind = kind; this.limit = count; this.replacing = replacing;
        this.selected.clear(); this.$('input[type="search"]').value = ""; this.path = ""; this.page = 0;
        this.$(".dialog-heading strong").textContent = `${replacing ? "Replace" : "Add"} ${kind}${count > 1 ? "s" : ""}`;
        this.dialog.setAttribute("aria-label", this.$(".dialog-heading strong").textContent);
        this.$('input[type="file"]').accept = `${kind}/*`;
        this.$('input[type="file"]').multiple = count > 1;
        this.$(".picker-apply").textContent = replacing ? "Replace" : "Add selected";
        this.dialog.showModal(); this.tab(this.source);
        return new Promise(resolve => { this.resolve = resolve; });
    }
    status(text, error = false) { this.$(".picker-status").textContent = text; this.$(".picker-status").classList.toggle("error", error); }
    update() {
        this.$(".picker-selection").textContent = `${this.selected.size} / ${this.limit} selected`;
        this.$(".picker-apply").disabled = !this.selected.size || this.loadingFiles;
        for (const button of this.dialog.querySelectorAll("[data-file-key]")) button.setAttribute("aria-pressed", String(this.selected.has(button.dataset.fileKey)));
    }
    tab(source) {
        this.abort?.abort(); this.source = source; this.path = ""; this.page = 0;
        this.$(".picker-upload").hidden = source !== "upload";
        this.$(".picker-browser").hidden = source === "upload";
        for (const tab of this.dialog.querySelectorAll("[data-source]")) {
            tab.setAttribute("aria-selected", String(tab.dataset.source === source));
            tab.tabIndex = tab.dataset.source === source ? 0 : -1;
        }
        this.status(`Select up to ${this.limit} ${this.kind}${this.limit > 1 ? "s" : ""}.`); this.update();
        if (source !== "upload") this.load();
    }
    upload(files) {
        if (this.loadingFiles) return;
        const items = [...files];
        if (items.length > this.limit) return this.status(`Select at most ${this.limit} files.`, true);
        if (items.some(file => file.type && !file.type.startsWith(`${this.kind}/`) && file.type !== "application/octet-stream")) return this.status(`Select ${this.kind} files.`, true);
        this.selected.clear();
        for (const [i, file] of items.entries()) this.selected.set(`upload:${i}`, {file});
        this.status(items.map(file => file.name).join(" · ")); this.update();
    }
    async load() {
        this.videoObserver.disconnect();
        this.abort?.abort(); this.abort = new AbortController();
        const signal = this.abort.signal;
        const grid = this.$(".browser-grid"); grid.replaceChildren();
        this.$(".browser-path").textContent = this.path || "/";
        this.$(".folder-up").disabled = !this.path;
        this.$(".page-prev").disabled = this.$(".page-next").disabled = true;
        this.status("Loading files…");
        try {
            const query = new URLSearchParams({source: this.source, path: this.path, kind: this.kind, search: this.$('input[type="search"]').value, page: this.page});
            const response = await api.fetchApi(`${BASE}/browse?${query}`, {signal});
            const data = await response.json();
            if (!response.ok) throw Error(data.error || "Could not load files.");
            if (signal.aborted) return;
            for (const entry of data.entries) {
                const button = document.createElement("button"); button.type = "button"; button.className = "browser-file"; button.title = entry.name;
                const label = document.createElement("span"); label.textContent = entry.name;
                if (entry.folder) {
                    button.append(document.createTextNode("📁"), label);
                    button.onclick = () => { this.path = entry.path; this.page = 0; this.$('input[type="search"]').value = ""; this.load(); };
                } else {
                    const source = this.source, key = `${source}:${entry.path}`;
                    const url = api.apiURL(`${BASE}/browse-file?${new URLSearchParams({source, path: entry.path})}`);
                    if (this.kind === "image") {
                        const img = document.createElement("img"); img.src = url; img.alt = ""; img.loading = "lazy"; button.append(img);
                    } else if (this.kind === "video") {
                        const video = document.createElement("video"); video.dataset.src = url; video.preload = "metadata"; video.muted = true; video.playsInline = true;
                        video.onloadedmetadata = () => { if (Number.isFinite(video.duration)) video.currentTime = Math.min(.1, video.duration / 2); };
                        button.append(video); this.videoObserver.observe(video);
                    } else button.append(document.createTextNode("♪"));
                    button.append(label); button.dataset.fileKey = key;
                    button.onclick = () => {
                        if (this.loadingFiles) return;
                        if (this.selected.has(key)) this.selected.delete(key);
                        else {
                            if (this.limit === 1) this.selected.clear();
                            if (this.selected.size >= this.limit) return this.status(`Select at most ${this.limit} files.`, true);
                            this.selected.set(key, {url, name: entry.name});
                        }
                        this.status( `${this.selected.size} selected. Click ${this.replacing ? "Replace" : "Add selected"} to use them.`); this.update();
                    };
                }
                grid.append(button);
            }
            this.$(".page-prev").disabled = this.page === 0;
            this.$(".page-next").disabled = (this.page + 1) * 100 >= data.total;
            this.$(".browser-pages span").textContent = `${this.page + 1} / ${Math.max(1, Math.ceil(data.total / 100))}`;
            this.status(data.total ? `${data.total} matching files and folders` : "No matching media in this folder."); this.update();
        } catch (error) { if (error.name !== "AbortError") this.status(error.message, true); }
    }
    async apply() {
        if (this.loadingFiles) return;
        this.loadingFiles = true; this.update(); this.status("Loading selected media…");
        this.downloadAbort = new AbortController();
        const signal = this.downloadAbort.signal;
        const controls = [...this.dialog.querySelectorAll("button, input")].filter(control => !control.matches(".picker-close, .picker-cancel"));
        const disabled = controls.map(control => control.disabled);
        for (const control of controls) control.disabled = true;
        const selection = [...this.selected.values()];
        try {
            const files = [];
            for (const item of selection) {
                if (item.file) files.push(item.file);
                else {
                    const response = await fetch(item.url, {signal});
                    if (!response.ok) throw Error(`Could not read ${item.name}. Refresh the folder and try again.`);
                    files.push(new File([await response.blob()], item.name, {type: response.headers.get("Content-Type") || ""}));
                }
            }
            if (!signal.aborted) this.close(files);
        } catch (error) { if (error.name !== "AbortError") this.status(error.message, true); }
        finally {
            controls.forEach((control, index) => { control.disabled = disabled[index]; });
            this.loadingFiles = false; this.update();
        }
    }
    close(files) {
        clearTimeout(this.searchTimer); this.abort?.abort(); this.downloadAbort?.abort();
        this.videoObserver.disconnect();
        for (const video of this.dialog.querySelectorAll("video")) { video.removeAttribute("src"); video.load(); }
        this.dialog.close(); this.resolve?.(files); this.resolve = null;
    }
    dispose() { this.close(null); this.dialog.remove(); }
}
