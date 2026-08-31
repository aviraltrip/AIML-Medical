import asyncio
import pytest
from pulsepoint_ai.engines.connect.translation import translator, translate_medical_text


def test_empty_string_translation():
    async def _test():
        res = await translator.translate("", target_lang="hi")
        assert res == ""
    asyncio.run(_test())


def test_translate_medical_text_wrapper():
    async def _test():
        text = "Take two tablets daily after food."
        res = await translate_medical_text(text, target_language="hi")
        assert isinstance(res, str)
        assert len(res) > 0
    asyncio.run(_test())
