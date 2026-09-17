from .two_pass import MiniMaxH3TwoPassSampler

NODE_CLASS_MAPPINGS = {"MiniMaxH3TwoPassSampler": MiniMaxH3TwoPassSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TwoPassSampler": "MiniMax H3 · Two Pass"}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
