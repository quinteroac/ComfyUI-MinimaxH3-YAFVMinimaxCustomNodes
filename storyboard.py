"""General-purpose storyboard prompt builder."""


_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight",
}


def build_storyboard_prompt(scene_count=4, general_scene_description="", style="", *scene_prompts):
    """Build the storyboard prompt without model enhancement or extra prose."""
    count = max(1, min(8, int(scene_count)))
    description = str(general_scene_description or "").strip() or "[general scene description]"
    aesthetic = str(style or "").strip() or "[style]"
    lines = [f"A {_NUMBER_WORDS[count]}-scene of {description}", ""]
    for index in range(count):
        prompt = str(scene_prompts[index] if index < len(scene_prompts) else "").strip()
        lines.append(f"Scene {index + 1}: {prompt or f'[Scene{index + 1} user prompt]'}")
        lines.append("")
    lines.append(
        f"Clean storyboard panel borders, professional {aesthetic} storyboard sheet, "
        "cinematic framing notes, coherent visual storytelling, highly detailed "
        f"{aesthetic} aesthetic, consistent characters across all six scenes."
    )
    return "\n".join(lines)


class YAFVStoryboardPrompt:
    CATEGORY = "YAFV/storyboard"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_prompt",)
    DESCRIPTION = "Create an editable, structured storyboard prompt without LLM enhancement."

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "scene_count": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            "general_scene_description": ("STRING", {
                "default": "", "multiline": True,
                "dynamicPrompts": False,
                "tooltip": "Overall story or setting shared by all storyboard panels.",
            }),
            "style": ("STRING", {"default": "cinematic", "multiline": False}),
        }
        for index in range(1, 9):
            required[f"scene_{index}"] = ("STRING", {
                "default": "", "multiline": True, "dynamicPrompts": False,
            })
        required["edited_prompt"] = ("STRING", {
            "default": "", "multiline": True, "dynamicPrompts": False,
            "tooltip": "Optional manual override from the storyboard editor.",
        })
        return {"required": required}

    def execute(self, scene_count, general_scene_description, style,
                scene_1="", scene_2="", scene_3="", scene_4="", scene_5="",
                scene_6="", scene_7="", scene_8="", edited_prompt=""):
        generated = build_storyboard_prompt(
            scene_count, general_scene_description, style,
            scene_1, scene_2, scene_3, scene_4, scene_5, scene_6, scene_7, scene_8,
        )
        return (str(edited_prompt).strip() or generated,)
