import unittest


class ExportProfilesTests(unittest.TestCase):
    def test_profile_sets_int8_recipes(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--profile",
                "litert-community-int8",
                "--model-path",
                "models/huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(ns.quantization_recipe, "dynamic_wi8_afp32")
        self.assertEqual(ns.vision_encoder_quantization_recipe, "weight_only_wi8_afp32")

    def test_late_quantization_recipe_overrides_profile(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--profile",
                "litert-community-int8",
                "--model-path",
                "models/huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
                "--output-dir",
                "out",
                "--quantization-recipe",
                "dynamic_wi4_afp32",
            ]
        )
        self.assertEqual(ns.quantization_recipe, "dynamic_wi4_afp32")

    def test_profile_does_not_enable_skip_quant_by_default(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--profile",
                "litert-community-int8",
                "--model-path",
                "models/huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
                "--output-dir",
                "out",
            ]
        )
        self.assertFalse(ns.skip_per_layer_embedder_quant)
        self.assertFalse(ns.skip_prefill_decode_quant)
        self.assertFalse(ns.ram_poor_export)

    def test_skip_per_layer_export_flag_disables_single_token_embedder(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--profile",
                "litert-community-int8",
                "--skip-per-layer-embedder-export",
                "--model-path",
                "models/huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
                "--output-dir",
                "out",
            ]
        )
        self.assertTrue(ns.skip_per_layer_embedder_export)
        self.assertFalse(export_gemma4._single_token_embedder_enabled(ns))

    def test_behavior_parity_mode_rejects_skip_flags(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--behavior-parity-mode",
                "--skip-per-layer-embedder-export",
                "--model-path",
                "models/huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
                "--output-dir",
                "out",
            ]
        )
        with self.assertRaises(SystemExit):
            export_gemma4._validate_behavior_parity_mode(ns)


if __name__ == "__main__":
    unittest.main()

