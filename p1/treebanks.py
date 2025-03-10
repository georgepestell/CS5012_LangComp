from io import open

from conllu import parse_incr

languages = ['en', 'orv', 'tr']

treebank = {}
treebank['en'] = 'UD_English-Atis/en_atis'
treebank['orv'] = 'UD_Old_East_Slavic-TOROT/orv_torot'
treebank['tr'] = 'UD_Turkish-Atis/tr_atis'

def train_corpus(lang):
	return treebank[lang] + '-ud-train.conllu'

def test_corpus(lang):
	return treebank[lang] + '-ud-test.conllu'

def conllu_corpus(path):
	data_file = open(path, 'r', encoding='utf-8')
	return list(parse_incr(data_file))
