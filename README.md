# ComfyUI-MinimaxH3-YAFVMinimaxCustomNodes - Yes another fucking vibecoded minimax H3 custom nodes for ComfyUI

I need to get profit from the codex subscription 

## YAFV · Editor multimedia

Restart ComfyUI and reload the browser after installing/updating this pack. Add
**YAFV → media → YAFV · Editor multimedia**. The node contains the queue/results,
preview, pencil and timeline together; it does not need a connection or a queued
execution. **Ampliar** enlarges the same editor, including the results panel.

- Drag an image/video from the results into the timeline, or double-click it.
  Images start at 3 seconds; drag clip edges or edit Entrada/Salida to change the
  duration. Drag clips to reorder; use Dividir, Duplicar and Eliminar clip.
- Select **Lápiz** and draw on the preview. Choose color, width, opacity and
  **Clip completo**, **Frame actual** or **Intervalo**. Interval times are relative
  to the start of the trimmed clip. The amber lane shows drawing intervals.
  Undo/redo also covers clip edits. Drawings remain fixed in image coordinates;
  there is no motion tracking.
- **Frame → timeline** adds the current frame as a still image, keeping its link
  to the original history entry. **Descargar PNG** downloads the current image
  or frame with drawings at source resolution. **Exportar MP4** downloads the
  sequence at the chosen size/FPS, fitting images with black borders and retaining
  source audio. Image clips and silent videos receive silence.
- **Cancelar** stops only the editor export, not ComfyUI generation.

Only output/temp media referenced by ComfyUI's in-memory `/history` are listed,
including native previews, VHS `gifs` results and this pack's `h3_preview`.
Pending/running counts come from the queue. The editor does not scan output
folders. Deleting history makes dependent clips unavailable. Switching workflows keeps each editor’s clips, drawings and undo history in browser
memory, keyed by graph and node. Returning to the workflow restores its timeline.
Reloading the page discards these sessions; edits are not saved in the workflow,
localStorage or a project file. Exports use temporary
files which are removed after completion/cancellation; original files are intact.

Video processing requires `ffmpeg` (with libx264) and `ffprobe` on PATH. No extra
Python package is needed beyond ComfyUI's Pillow/aiohttp. Browser preview support
depends on the source codec; MP4/H.264 is recommended. GIF/MKV/AVI sources may be
processable by FFmpeg but unplayable in the browser. There are no transitions,
multiple video tracks or animated brush interpolation in this version.

Tests: `python -m unittest discover -s tests -v` using ComfyUI's Python environment.
The editor tests use synthetic media and an isolated HTTP server, without changing
ComfyUI's queue or loading models.
