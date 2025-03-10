from treebanks import languages, train_corpus, test_corpus, conllu_corpus

if __name__ == '__main__':
	for lang in languages:
		train_sents = conllu_corpus(train_corpus(lang))
		test_sents = conllu_corpus(test_corpus(lang))

		# Illustration how to access the word and the part-of-speech of tokens.
		for sent in train_sents:
			for token in sent:
				print(token['form'], '->', token['upos'], sep='', end=' ')
			print()

		print('language', lang)
		print(len(train_sents), 'training sentences')
		print(len(test_sents), 'test sentences')
