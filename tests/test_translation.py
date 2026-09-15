import asyncio

from pulsepoint_ai.engines.connect.translation import translate_medical_text, translator


def test_empty_string_translation() -> None:
    async def _test() -> None:
        res = await translator.translate("", target_lang="hi")
        assert res == ""
    asyncio.run(_test())


def test_translate_medical_text_wrapper() -> None:
    async def _test() -> None:
        text = "Take two tablets daily after food."
        res = await translate_medical_text(text, target_language="hi")
        assert isinstance(res, str)
        assert len(res) > 0
    asyncio.run(_test())
