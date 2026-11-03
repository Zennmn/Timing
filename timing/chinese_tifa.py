"""Native Mandarin pronunciation candidates for the TIFA worker."""
import re
from .chinese import HAN, chinese_units, pinyin_groups, phrase_readings

CONVERTER_ID = 'timing-mandarin'


def ensure_registered():
    from g2pflow.registry import converter, get_converter, list_converters
    if CONVERTER_ID in list_converters():
        return get_converter(CONVERTER_ID)
    from g2pflow.converters.base import Converter, G2PWord, G2PGroup, G2PReading
    class MandarinConverter(Converter):
        def __init__(self, overrides=None):
            self.overrides = overrides or {}

        def find(self, text):
            match = re.search('(?:'+HAN+'|[0-9])+', text)
            return match.span() if match else None

        def _convert(self, text):
            from pypinyin import pinyin, Style
            words = []
            units = chinese_units(text)
            phrases = phrase_readings(units, self.overrides)
            for index, unit in enumerate(units):
                custom = self.overrides.get(unit.spoken) or phrases.get(index)
                choices = [custom] if custom else [unit.reading]
                if not custom and len(unit.spoken) == 1:
                    choices += [reading for reading in pinyin(unit.spoken, style=Style.NORMAL, heteronym=True,
                                                               strict=False)[0] if reading not in choices]
                readings = []
                for reading in choices:
                    try:
                        groups = [G2PGroup(group['script'], group['phones']) for group in pinyin_groups(reading)]
                    except ValueError:
                        if custom: raise
                        continue
                    readings.append(G2PReading(paths=[groups]))
                if not readings:
                    raise ValueError(f'「{unit.spoken}」的默认读音不在中文声学词典中，请补充自定义读音。')
                words.append(G2PWord(unit.spoken, readings=readings))
            return words

    converter(id=CONVERTER_ID, language='zh')(MandarinConverter)
    return MandarinConverter
