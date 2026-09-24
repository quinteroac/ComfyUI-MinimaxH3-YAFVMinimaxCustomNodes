# YAFV interactive project templates

The interactive templates generate and approve one segment at a time using
the MiniMax H3 nodes.

1. Open the Ref2Vid or FL2V template.
2. Set the same `project_name` and the current `segment_index` on
   `YAFV Project Context` and `YAFV Project Commit`.
3. Generate one candidate; `YAFV Project Review` waits for your decision.
4. Approve to commit and update the timeline, or reject without committing.
5. Increment `segment_index` and queue the next candidate.

Continuity stores both sampling passes in a `.safetensors` file beside each
approved MP4. The templates connect:

- Two Pass `latent_pass1` and `latent` → Commit `latent_pass1` and `latent_final`.
- Motion Context `trim_frames` → both Media Trim and Commit.
- Project Context `context_latent` → Motion Context (Pass 1 resolution).
- Project Context `context_latent_pass2` → Two Pass (final resolution).
- Project Context `output_frames` → Media Trim; `generation_length` → H3 length.

Pass 2 restores the previous final-resolution video/audio prefix after upscale,
then freezes it during refinement. Set `pass1_return_with_leftover_noise=False`.
Keep both pass resolutions fixed within a scene. Context loads the latents
associated with the approved generation, without VAE re-encoding. Existing
MP4-only generations still use RGB extraction and VAE encoding for Pass 1;
their Pass 2 context is unavailable until a new generation saves both latents.

Set Project Context `target_width`/`target_height` to Pass 1 dimensions, and
match its `context_frames` to Motion Context `context_length`. Continuations
round the requested delivered length up to a multiple of 17, so the delivered
video ends at the latent's final frame. For 124 requested frames and 22 context
frames: generate 158, trim 22, deliver **136**. The initial clip delivers 124.
Commit checks the actual MP4 length and resolution against the saved latents;
apply pixel upscale, interpolation or further editing after Commit.

`YAFV Project 2 Dual Latent.json` is the executed project_2 workflow with these
connections added. It starts at segment 0 under `project_2_latent_test`, using
0.2 MP → 0.5 MP and the original model/prompt settings. Approve the first clip,
then increment the segment index to test direct latent continuation.

Project Context's `scene_mode` defaults to **Continue scene**. Select
**New scene** for a scene cut at any segment index: image, last-frame,
video and latent context outputs become `None` (the path is an empty string).
Even `initial_frame` is ignored in this mode. Generation length equals the
H3-aligned clip length, and Motion Context returns zero trim frames.
Prompts and creative references stay in their existing nodes. Project name,
segment numbering, review and timeline composition are unaffected.
Keep this selection when retrying the new scene; for its following clip,
select **Continue scene** to use the newly approved segment as context.
