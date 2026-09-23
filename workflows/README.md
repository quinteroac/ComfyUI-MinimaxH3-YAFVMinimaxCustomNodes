# YAFV interactive project templates

The two JSON files are segment-by-segment templates built around the existing
MiniMax H3 nodes.

1. Open the Ref2Vid or FL2V template.
2. Set the same `project_name` and the current `segment_index` on
   `YAFV Project Context` and `YAFV Project Commit`.
3. Generate one candidate; `YAFV Project Review` waits for your decision.
4. Approve to commit and update the timeline, or reject without committing.
5. Increment `segment_index` and queue the next candidate.

Continuity uses native H3 context latents and synchronized media trimming.
The previous clip is not inserted into creative `ref_video` slots.

Project Context's `scene_mode` defaults to **Continue scene**. Select
**New scene** for a scene cut at any segment index: image, last-frame,
video and latent context outputs become `None` (the path is an empty string).
Even `initial_frame` is ignored in this mode. Generation length equals the
H3-aligned clip length, and Motion Context returns zero trim frames.
Prompts and creative references stay in their existing nodes. Project name,
segment numbering, review and timeline composition are unaffected.
Keep this selection when retrying the new scene; for its following clip,
select **Continue scene** to use the newly approved segment as context.
