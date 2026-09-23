import unittest

from storyboard import YAFVStoryboardPrompt, build_storyboard_prompt


class StoryboardPromptTests(unittest.TestCase):
    def test_default_structure_uses_four_scenes(self):
        prompt = build_storyboard_prompt(4, "A train crossing the desert", "watercolor", "Wide shot")
        self.assertTrue(prompt.startswith("A four-scene of A train crossing the desert\n\n"))
        self.assertIn("Scene 1: Wide shot", prompt)
        self.assertIn("Scene 4: [Scene4 user prompt]", prompt)
        self.assertTrue(prompt.endswith("consistent characters across all six scenes."))

    def test_count_is_clamped_and_output_can_be_edited(self):
        prompt = build_storyboard_prompt(99, "Story", "comic", *(["beat"] * 8))
        self.assertIn("A eight-scene of Story", prompt)
        self.assertNotIn("Scene 9:", prompt)
        output = YAFVStoryboardPrompt().execute(2, "Story", "comic", "One", "Two", edited_prompt="Custom")
        self.assertEqual(output, ("Custom",))


if __name__ == "__main__":
    unittest.main()
