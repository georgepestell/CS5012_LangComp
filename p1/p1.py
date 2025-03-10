# Estimate transition probabilities and the emission probabilities of an HMM, on the basis of (tagged) sentences from a training corpus from Universal Dependencies. Includes start-of-sentence  and end-of-sentence markers in estimation.

DEBUG=True

from treebanks import languages, train_corpus, test_corpus, conllu_corpus

if __name__ == '__main__':
    for lang in languages:

        # Just do english for now
        if lang != 'en':
            continue

        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))

        # Dictionary to store transition and emission counts
        transitions = {}
        emissions = {}

        # Loop over each sentence in the training corpus
        for sentence in train_sents:
            # Begin with a start marker
            prev_token = '<s>'

            for word in sentence:
                curr_token = word["upos"]
                curr_word = word["form"].lower() # Convert to lowercase to avoid case sensitivity

                ### Update emissions
                # Add token to emissions
                if (curr_token not in emissions):
                    emissions[curr_token] = {}

                # Add word to token emissions
                if (curr_word not in emissions[curr_token]):
                    emissions[curr_token][curr_word] = 0

                # Increment count for word-token emission
                emissions[curr_token][curr_word] += 1

                ### Update transitions

                # Add previous token to transitions
                if (prev_token not in transitions):
                    transitions[prev_token] = {}

                # Add current token to transitions for previous token
                if (curr_token not in transitions[prev_token]):
                    transitions[prev_token][curr_token] = 0
                
                # Increment count for transition from previous to current token
                transitions[prev_token][curr_token] += 1

                # Update previous token
                prev_token = curr_token

            # Add end marker
            if (prev_token not in transitions):
                transitions[prev_token] = {}
            if ('</s>' not in transitions[prev_token]):
                transitions[prev_token]['</s>'] = 0
            transitions[prev_token]['</s>'] += 1


        if (DEBUG):
            # Print transition counts
            for prev_pos in transitions:
                for curr_pos in transitions[prev_pos]:
                    print(f"{prev_pos} -> {curr_pos}: {transitions[prev_pos][curr_pos]}")            

            # Print emission counts
            for pos in emissions:
                for word in emissions[pos]:
                    print(f"{pos} -> {word}: {emissions[pos][word]}")

        # Possible unseen tags z
        z = 100000

        # Get all unique words 
        all_words = set()
        for pos in emissions:
            all_words.update(emissions[pos].keys())

        # Smoothed probabilities
        smoothed_emissions = {}

        # Calculate probabilities for seen and unseen words
        for pos in emissions:
            smoothed_emissions[pos] = {}

            n = sum(emissions[pos].values())
            m = len(emissions[pos])

            for word in all_words:
                if word in emissions[pos]:
                    smoothed_emissions[pos][word] = emissions[pos][word] / (n + m)
                else:
                    smoothed_emissions[pos][word] = m / (z * (n + m))
        
        # Calculate transition probabilities with bigram 

        all_pos_tags = set(['<s>', '</s>'])
        for pos in emissions:
            all_pos_tags.update(pos)

        smoothed_transitions = {}

        for prev_pos in transitions:
            smoothed_transitions[prev_pos] = {}

            n = sum(transitions[prev_pos].values())
            m = len(transitions[prev_pos])

            for pos in all_pos_tags:
                if pos in transitions[prev_pos]:
                    smoothed_transitions[prev_pos][pos] = transitions[prev_pos][pos] / (n + m)
                else:
                    smoothed_transitions[prev_pos][pos] = m / (z * (n + m))

        # Print smoothed probabilities
        if DEBUG:
            print("Smoothed emission probabilities")
            for pos in smoothed_emissions:
                for word in smoothed_emissions[pos]:
                    print(f"{pos} -> {word}: {smoothed_emissions[pos][word]}")
            print("Smoothed transition probabilities")
            for prev_pos in smoothed_transitions:
                for pos in smoothed_transitions[prev_pos]:
                    print(f"{prev_pos} -> {pos}: {smoothed_transitions[prev_pos][pos]}")  