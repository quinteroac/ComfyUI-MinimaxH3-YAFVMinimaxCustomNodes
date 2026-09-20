from .two_pass import MiniMaxH3TwoPassSampler
from .media_editor import YAFVMediaEditor
from .video_prompts import YAFVVideoPrompts
from .reference_prompts import YAFVReferenceVideoPrompts
from .h3_extend import YAFVH3VideoExtend, YAFVH3EncodeAV

NODE_CLASS_MAPPINGS = {"MiniMaxH3TwoPassSampler": MiniMaxH3TwoPassSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TwoPassSampler": "MiniMax H3 · Two Pass"}
NODE_CLASS_MAPPINGS["YAFVMediaEditor"] = YAFVMediaEditor
NODE_DISPLAY_NAME_MAPPINGS["YAFVMediaEditor"] = "YAFV · Editor multimedia"
NODE_CLASS_MAPPINGS["YAFVVideoPrompts"] = YAFVVideoPrompts
NODE_DISPLAY_NAME_MAPPINGS["YAFVVideoPrompts"] = "YAFV · Prompts para video"
NODE_CLASS_MAPPINGS["YAFVReferenceVideoPrompts"] = YAFVReferenceVideoPrompts
NODE_DISPLAY_NAME_MAPPINGS["YAFVReferenceVideoPrompts"] = "YAFV · Prompts para Reference to Video"
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS.update({
    "YAFVH3VideoExtend": YAFVH3VideoExtend,
    "YAFVH3EncodeAV": YAFVH3EncodeAV,
    "MiniMaxH3VideoExtendPatched": YAFVH3VideoExtend,
    "MiniMaxH3EncodeAVPatched": YAFVH3EncodeAV,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "YAFVH3VideoExtend": "YAFV · H3 Video Extend",
    "YAFVH3EncodeAV": "YAFV · H3 Encode AV",
    "MiniMaxH3VideoExtendPatched": "YAFV · H3 Video Extend (legacy workflow)",
    "MiniMaxH3EncodeAVPatched": "YAFV · H3 Encode AV (legacy workflow)",
})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
