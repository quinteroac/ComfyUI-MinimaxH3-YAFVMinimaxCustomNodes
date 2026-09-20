# ComfyUI-MinimaxH3-YAFVMinimaxCustomNodes - Yes another fucking vibecoded minimax H3 custom nodes for ComfyUI

I need to get profit from the codex subscription 

## YAFV · H3 Video Extend / Encode AV

This pack provides H3 continuation directly, with no H3-Extend dependency or
global model patches. It requires native ComfyUI H3 support for arbitrary
video/audio keyframe positions. Disable the old **ComfyUI-MiniMax-H3-Extend**
pack and restart ComfyUI: its startup patch replaces that native support.

Connect a previous H3 latent to **YAFV · H3 Video Extend**, then connect its
positive conditioning and new latent to Two Pass (or a native sampler).
For external footage, **YAFV · H3 Encode AV** encodes video at 24 fps and optional
audio into a context latent. Continuation inherits the source resolution.
`context_frames` counts trailing latent positions; `length` counts new video
frames. Video context ends at frame zero; audio context ends just before it.
The node supports first/last images, automatic last-frame pinning, batched
reference images, reference video and reference audio. Audio references require
`audio_vae`. Temporal Pass 2 preserves context at each window's local origin
and resizes its video guides to the window canvas.

Existing workflows using `MiniMaxH3VideoExtendPatched` or
`MiniMaxH3EncodeAVPatched` load through local aliases. This pack does not inject
classes into ComfyUI's native namespace; third-party nodes relying on H3-Extend's
namespace injection must be replaced with the local Video Extend node.
Reference preparation was adapted from kat3ri/ComfyUI-MiniMax-H3-Extend
(declared MIT) and the native ComfyUI nodes.

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

## YAFV · Prompts para video

Add **YAFV → video → YAFV · Prompts para video**. Connect `generated_prompt`
(STRING), `first_frame` (IMAGE) and `last_frame` (IMAGE) to your image-to-video
node. Missing frames return `None`; connect them to optional image inputs that
accept missing images, such as MiniMax H3 Image to Video.

Each list element contains a prompt and optional first/last images. Use **Nuevo**,
load images by dropping files or clicking their previews, then **Agregar**.
Select an element to execute it with ComfyUI's normal queue. **Guardar cambios**
updates an element; the remove controls delete an element or one of its images.
Unsaved edits are drafts: the queue uses the saved selection, shown in the footer.
Switching selections offers save/discard/keep-editing controls.

Optional inputs:

- `clip`: the same CLIP model object accepted by ComfyUI's **Generate Text**.
- `text`: system instructions. When both CLIP and nonempty instructions are
  supplied, this node calls the native `TextGenerate.execute` path. Otherwise it
  returns the saved prompt verbatim, without generating text.

Instructions and the selected prompt are sent as separate text sections through
Generate Text's model template, not as a separate system-role API. Reference
images require a visual model. A single frame is identified as first or last;
with both frames, the model sees a labeled side-by-side reference. Output images
keep their original size; EXIF orientation is normalized and transparency is
composited onto white for RGB IMAGE outputs. Model errors are reported instead of
silently substituting the original prompt. The advanced section exposes native
text-generation settings (512-token limit, sampling off, thinking off, native
template on and MTP auto by default).

Lists, uploaded images and generated results live only in server RAM, separately
for each graph/node. They survive browser reloads and workflow switches, and are
cleared when ComfyUI restarts. Workflow files contain only library/revision
identifiers and generation settings, not list content or uploaded images. Saving
a copy with the same graph/node identifiers refers to the same session library.
Editing creates an immutable revision; queued jobs keep their original text and
frames even after an edit/deletion. Unused revisions are released on subsequent
library access once submission and queued/running jobs no longer reference them.
Removing a node does not delete its server library before the session ends.

Restart ComfyUI once after installing this node and reload the browser. No new
Python dependencies are required. Tests exercise the native Generate Text path
with a fake CLIP, without loading or downloading model weights.

## YAFV · Prompts para Reference to Video

Add **YAFV → video → YAFV · Prompts para Reference to Video**. It uses the same
session-only prompt list, `clip`/`text` inputs and Generate Text settings as
**Prompts para video**. Each entry can hold up to eight images, two videos with
independent soundtracks, and three standalone audio files. Use **+ Añadir
referencia** and choose Imagen, Video or Audio. A card appears only after a file
is selected; canceling leaves the panel unchanged. Existing cards accept dropped
replacement files. Each video contains its soundtrack controls. Removing a card
frees its slot without renumbering other references. Save the entry before queuing.

Connect `generated_prompt` to MiniMax H3 Reference to Video's `prompt`, and the
reference outputs to its matching optional inputs: `ref_image_0`–`ref_image_7`,
`ref_video_0`–`ref_video_1`, `ref_video_audio_0`–`ref_video_audio_1`, and
`ref_audio_0`–`ref_audio_2`. The original eight output positions are preserved;
new outputs are appended, for 16 total. Output sockets remain available as cards
are added or removed. Video outputs are IMAGE batches at 24 fps, not VIDEO sockets.
Images retain their dimensions; video retains its dimensions and duration to the
nearest frame. H3 performs its own frame-count alignment. Missing references
return `None`.

Loading/replacing a video extracts its soundtrack automatically. An explicit
audio upload or removal in the same save takes priority; silent videos have no
soundtrack output. Removing the video also removes its automatic soundtrack.
Standalone audio files are not analyzed by the prompt generator. With `clip`
and nonempty `text`, the generator sees a labeled sheet of the attached images and
up to eight evenly spaced frames from each video, plus the mapping of present references
to MiniMax's `<Picture i>`, `<Video k>` and `<Audio j>` tags. Connect outputs to
the matching slots to preserve this ordering. Without both inputs, the original
prompt is returned unchanged.

FFmpeg with libx264 is required. Uploaded references live in temporary session
files, not in the workflow. Editing or deleting an entry preserves any revision
already submitted to the queue; unreferenced files are removed on subsequent
library activity. Restarting ComfyUI clears the list. Browser playback depends
on codec support; generation uses the saved, decoded references.
