import unittest
from argparse import Namespace


class ExportProfilesTests(unittest.TestCase):
    def test_profile_sets_int8_recipes(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--profile",
                "litert-community-int8",
                "--model-path",
                "models/example-gemma4",
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
                "models/example-gemma4",
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
                "models/example-gemma4",
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
                "models/example-gemma4",
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
                "models/example-gemma4",
                "--output-dir",
                "out",
            ]
        )
        with self.assertRaises(SystemExit):
            export_gemma4._validate_behavior_parity_mode(ns)

    def test_backend_registry_has_gemma4(self) -> None:
        from safetensors_to_litertlm.converter.backends.registry import get_backend

        backend = get_backend("gemma4")
        self.assertEqual(backend.key, "gemma4")

    def test_auto_fallback_text_only_default_enabled(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--model-path",
                "models/example-gemma4",
                "--output-dir",
                "out",
            ]
        )
        self.assertTrue(ns.auto_fallback_text_only)
        self.assertEqual(ns.multimodal_intent, "legacy")

    def test_no_auto_fallback_text_only_override(self) -> None:
        from safetensors_to_litertlm.converter import export_gemma4

        ns = export_gemma4._parse_export_args(
            [
                "--no-auto-fallback-text-only",
                "--model-path",
                "models/example-gemma4",
                "--output-dir",
                "out",
            ]
        )
        self.assertFalse(ns.auto_fallback_text_only)

    def test_known_vision_error_enables_text_only_fallback(self) -> None:
        from safetensors_to_litertlm.converter.backends import gemma4

        args = Namespace(
            task="image_text_to_text",
            export_vision_encoder=True,
            auto_fallback_text_only=True,
        )
        exc = AttributeError(
            "'Gemma3ForConditionalGeneration' object has no attribute 'vision_tower'"
        )
        fallback = gemma4.maybe_prepare_text_only_fallback(args, exc)
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.task, "text_generation")
        self.assertFalse(fallback.export_vision_encoder)

    def test_known_vision_error_without_fallback_raises_system_exit(self) -> None:
        from safetensors_to_litertlm.converter.backends import gemma4

        args = Namespace(
            task="image_text_to_text",
            export_vision_encoder=True,
            auto_fallback_text_only=False,
        )
        exc = AttributeError(
            "'Gemma3ForConditionalGeneration' object has no attribute 'vision_tower'"
        )
        with self.assertRaises(SystemExit):
            gemma4.maybe_prepare_text_only_fallback(args, exc)

    def test_non_vision_error_does_not_trigger_fallback(self) -> None:
        from safetensors_to_litertlm.converter.backends import gemma4

        args = Namespace(
            task="image_text_to_text",
            export_vision_encoder=True,
            auto_fallback_text_only=True,
        )
        exc = RuntimeError("some unrelated export failure")
        fallback = gemma4.maybe_prepare_text_only_fallback(args, exc)
        self.assertIsNone(fallback)

    def test_planner_best_effort_downgrades_when_no_vision(self) -> None:
        from safetensors_to_litertlm.converter.backends import gemma4

        args = Namespace(
            multimodal_intent="best-effort",
            task="image_text_to_text",
            export_vision_encoder=True,
            auto_fallback_text_only=True,
        )
        caps = gemma4.ModelCapabilities(
            model_type="gemma3",
            architectures=("Gemma3ForConditionalGeneration",),
            has_vision_encoder=False,
            supports_multimodal=False,
            reason="no vision signals in config",
        )
        plan = gemma4.plan_export_mode(args, caps)
        self.assertEqual(plan.selected_task, "text_generation")
        self.assertFalse(plan.export_vision_encoder)
        self.assertEqual(plan.reason_code, "best_effort_downgrade_text_only")

    def test_planner_strict_raises_without_multimodal_support(self) -> None:
        from safetensors_to_litertlm.converter.backends import gemma4

        args = Namespace(
            multimodal_intent="strict",
            task="image_text_to_text",
            export_vision_encoder=True,
            auto_fallback_text_only=True,
        )
        caps = gemma4.ModelCapabilities(
            model_type="gemma3",
            architectures=("Gemma3ForConditionalGeneration",),
            has_vision_encoder=False,
            supports_multimodal=False,
            reason="no vision signals in config",
        )
        with self.assertRaises(SystemExit):
            gemma4.plan_export_mode(args, caps)

    def test_planner_text_only_intent_forces_text_generation(self) -> None:
        from safetensors_to_litertlm.converter.backends import gemma4

        args = Namespace(
            multimodal_intent="text-only",
            task="image_text_to_text",
            export_vision_encoder=True,
            auto_fallback_text_only=True,
        )
        caps = gemma4.ModelCapabilities(
            model_type="gemma3",
            architectures=("Gemma3ForConditionalGeneration",),
            has_vision_encoder=True,
            supports_multimodal=True,
            reason="vision signals detected",
        )
        plan = gemma4.plan_export_mode(args, caps)
        self.assertEqual(plan.selected_task, "text_generation")
        self.assertFalse(plan.export_vision_encoder)
        self.assertEqual(plan.reason_code, "intent_text_only")


if __name__ == "__main__":
    unittest.main()

