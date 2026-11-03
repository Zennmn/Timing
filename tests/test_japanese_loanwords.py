from pathlib import Path
import sys

import pytest

from timing.text import tokenize
from timing.ctc import text_units
from timing.singing import phonemize, japanese_parts
from timing.schema import Line
from timing.japanese_loanwords import japanese_loanword_spans


@pytest.mark.parametrize('surface,kana',[('Tシャツ','ティーシャツ'),('tシャツ','ティーシャツ'),
                                       ('Ｔシャツ','ティーシャツ'),('Yシャツ','ワイシャツ')])
def test_letter_compounds_use_japanese_in_mms_sofa_and_keep_original_surface(surface,kana):
    text = 'くれた' + surface + 'を'
    tokens = tokenize(text, fine=False)
    word = next(u for u in tokens if u.text == surface)
    assert word.reading == kana
    assert ''.join(t for t,_ in japanese_parts(word,True)) == surface
    for fine in (False,True):
        units,_,phones,_ = phonemize([Line(text)], fine=fine)
        assert ''.join(u.text for u in units) == text
        assert set(lang for lang,_ in phones) == {'ja','sp'}
    units,labels = text_units(text)
    assert ''.join(u.text for u in units) == text
    assert all('T' not in label and 'シャ' not in label for label in labels)


def test_compounds_do_not_claim_english_or_suffix_of_larger_latin_word():
    for text in ('T shirt', 'Tokyo', 'ATシャツ', 'myシャツ', 'Shirts'):
        assert list(japanese_loanword_spans(text)) == []


def test_native_compound_has_only_japanese_phones_and_display_uses_real_groups():
    model = Path(__file__).resolve().parents[1]/'models/tifa'
    if not (model/'deps/g2pflow').exists():
        pytest.skip('TIFA runtime not downloaded')
    sys.path.insert(0,str(model/'deps'))
    from timing.tifa_pronunciation import ensure_loanwords_registered
    from timing.tifa_worker import display_plans
    converter = ensure_loanwords_registered()(dict_path=str(model/'dictionaries/japanese_dict_full.txt'))
    assert converter.find('くれたtシャツ') == (3,7)
    assert converter.find('T shirt') is None
    word = converter.convert('tシャツ')[0]
    groups = word.readings[0].paths[0]
    choice = {'scripts':[g.script for g in groups],
              'phonemes':['ja/'+p for g in groups for p in g.phonemes]}
    assert all(p.startswith('ja/') for p in choice['phonemes'])
    plan = display_plans(['Tシャツ'],[choice],model)[0]
    assert ''.join(x['text'] for x in plan) == 'Tシャツ'
    assert sum(x['group_count'] for x in plan) == len(groups)
    assert [x['text'] for x in plan] == ['T','シャ','ツ']
