from .two_pass import MiniMaxH3TwoPassSampler
from .media_editor import YAFVMediaEditor

NODE_CLASS_MAPPINGS = {"MiniMaxH3TwoPassSampler": MiniMaxH3TwoPassSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TwoPassSampler": "MiniMax H3 · Two Pass"}
NODE_CLASS_MAPPINGS["YAFVMediaEditor"] = YAFVMediaEditor
NODE_DISPLAY_NAME_MAPPINGS["YAFVMediaEditor"] = "YAFV · Editor multimedia"
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
