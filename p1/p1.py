# Estimate transition probabilities and the emission probabilities of an HMM, on the basis of (tagged) sentences from a training corpus from Universal Dependencies. Includes start-of-sentence  and end-of-sentence markers in estimation.

DEBUG=True

from treebanks import languages, train_corpus, test_corpus, conllu_corpus

from sys import float_info
import math
import numpy as np

class HMM:
    def __init__(self, train_sents, z=100000):
        self.transitions = {}
        self.emissions = {}
        self.vocab = set()
        self.state_counts = {}
        self.z = z
    
        self.transition_probs = {}
        self.emission_probs = {}
        self.train(train_sents)
    
    def calculateLambda(self, state):
        uniqueTransitions = len(self.transitions[state])
        if self.state_counts[state] == 0:
            return 0
        return self.state_counts[state] / (self.state_counts[state] + uniqueTransitions)
    

    def train(self, train_sents):
    # Loop over each sentence in the training corpus
        for sentence in train_sents:
            states = ["<s>"]
            words = ["<s>"]
            for token in sentence:
                states.append(token['upos'])
                words.append(token['form'])
                self.vocab.add(token['form'])
            states.append("</s>")
            words.append("</s>")

            for i in range(1, len(states)):
                prev_state, curr_state = states[i-1], states[i]

                if prev_state not in self.transitions:
                    self.transitions[prev_state] = {}

                if curr_state not in self.transitions[prev_state]:
                    self.transitions[prev_state][curr_state] = 0

                if prev_state not in self.state_counts:
                    self.state_counts[prev_state] = 0

                self.transitions[prev_state][curr_state] += 1
                self.state_counts[prev_state] += 1

            for state, word in zip(states, words):
                if state not in self.emissions:
                    self.emissions[state] = {}

                if word not in self.emissions[state]:
                    self.emissions[state][word] = 0

                self.emissions[state][word] += 1
        
        # Add the <s> and </s> states to the total states
        self.state_counts["<s>"] = len(train_sents)
        self.state_counts["</s>"] = len(train_sents)

        for prev_state in self.transitions:
            prev_lambda = self.calculateLambda(prev_state)
            total_transitions = sum(self.transitions[prev_state].values())

            if prev_state not in self.transition_probs:
                self.transition_probs[prev_state] = {}
            
            for curr_state in self.transitions[prev_state]:
                transition_prob = self.transitions[prev_state][curr_state] / total_transitions
                self.transition_probs[prev_state][curr_state] = (prev_lambda * transition_prob + (1 - prev_lambda) * (self.state_counts[curr_state] / sum(self.state_counts.values())))

            # Unseen transitions
            unseen_value = (1 - prev_lambda) * (1 - sum(self.transition_probs[prev_state].values()))
            self.transition_probs[prev_state]["<unk>"] = unseen_value
        
        for state in self.emissions:
            n = sum(self.emissions[state].values())
            m = len(self.emissions[state])

            for word in self.emissions[state]:
                if state not in self.emission_probs:
                    self.emission_probs[state] = {}
                self.emission_probs[state][word] = self.emissions[state][word] / (n + m)

            unseen_value = m / (self.z * (n + m))
            self.emission_probs[state]["<unk>"] = unseen_value
    
    def getEmissionProbility(self, state, word):
        if state in self.emission_probs and word in self.emission_probs[state]:
            return self.emission_probs[state][word]
        return self.emission_probs[state]["<unk>"]
    
    def getTransitionProbability(self, prev_state, curr_state):
        if (prev_state == "</s>"):
            return 0
        if prev_state in self.transition_probs and curr_state in self.transition_probs[prev_state]:
            return self.transition_probs[prev_state][curr_state]
        return self.transition_probs[prev_state]["<unk>"]
    
class Viterbi:
    def __init__(self, hmm):
        self.hmm = hmm
        self.states = list(hmm.state_counts.keys())

    def run(self, sentence):
        T = len(sentence)
        N = len(self.states)

        # Construct the 2-d table
        viterbi = np.zeros((N, T))
        backpointers = np.zeros((N, T), dtype=int)

        # First column
        for i, state in enumerate(self.states):
            viterbi[i, 0] = math.log(self.hmm.getTransitionProbability("<s>", state)) + math.log(self.hmm.getEmissionProbility(state, sentence[0]))
        
        # Rest of the columns
        for t in range(1, T):
            for i, state in enumerate(self.states):
                max_prob = -math.inf
                max_index = -1

                for j, prev_state in enumerate(self.states):
                    transition_prob = max(1e-10, self.hmm.getTransitionProbability(prev_state, state))
                    emission_prob = max(1e-10, self.hmm.getEmissionProbility(state, sentence[t]))
                    prob = viterbi[j, t-1] + math.log(transition_prob) + math.log(emission_prob)
                    if prob > max_prob:
                        max_prob = prob
                        max_index = j
                
                viterbi[i, t] = max_prob
                backpointers[i, t] = max_index
        
        # Find the best path
        best_path = []
        last_state_idx = np.argmax(viterbi[:, T-1])
        best_path.append(self.states[last_state_idx])

        for t in range(T-1, 0, -1):
            last_state_idx = backpointers[last_state_idx, t]
            best_path.append(self.states[last_state_idx])

        return list(reversed(best_path))

class LanguageModel:
    def __init__(self, hmm): 
        self.hmm = hmm
        self.min_log_prob = -float_info.max

    ### BEGIN STARTER CODE ###
    # Adding a list of probabilities represented as log probabilities.
    def logsumexp(self, vals):
        if len(vals) == 0:
            return self.min_log_prob
        m = max(vals)
        if m == self.min_log_prob:
            return self.min_log_prob
        else:
            return m + math.log(sum([math.exp(val - m) for val in vals]))
    ### END STARTER CODE ###

    def calculatePerplexity(self, test_sents):
        total_log_prob = 0
        total_words = 0
        for sentence in test_sents:
            words = ["<s>"] + [token['form'] for token in sentence] + ["</s>"]
            T = len(words)
            N = len(self.hmm.state_counts)
            forward = np.zeros((N, T))
            states = list(self.hmm.state_counts.keys())

            # Initialize first column (in log space)
            for i, state in enumerate(states):
                forward[i, 0] = math.log(max(1e-10, self.hmm.getTransitionProbability("<s>", state))) + \
                        math.log(max(1e-10, self.hmm.getEmissionProbility(state, words[0])))

            # Fill rest of the table (in log space)
            for t in range(1, T):
                for i, state in enumerate(states):
                    log_probs = []
                    for j, prev_state in enumerate(states):
                        trans_prob = max(1e-10, self.hmm.getTransitionProbability(prev_state, state))
                        log_probs.append(forward[j, t-1] + math.log(trans_prob))
                        
                    forward[i, t] = self.logsumexp(log_probs) + \
                        math.log(max(1e-10, self.hmm.getEmissionProbility(state, words[t])))

            # Sum final column for sentence probability (already in log space)
            sentence_log_prob = self.logsumexp(forward[:, T-1])
            total_log_prob += sentence_log_prob / math.log(2)  # convert to log base 2
            total_words += len(sentence)

        perplexity = math.pow(2, -total_log_prob / total_words)
        return perplexity

### BEGIN STARTER CODE ###
if __name__ == '__main__':
    for lang in languages:

        train_sents = conllu_corpus(train_corpus(lang))
        test_sents = conllu_corpus(test_corpus(lang))

        ### END STARTER CODE ###

        hmm = HMM(train_sents)
        viterbi = Viterbi(hmm)

        # Test accuracy
        correct = 0
        total = 0
        for sentence in test_sents:
            words = [token['form'] for token in sentence]
            tags = [token['upos'] for token in sentence]
            predicted_tags = viterbi.run(words)

            for i in range(len(tags)):
                if tags[i] == predicted_tags[i]:
                    correct += 1
                total += 1
        
        print(lang, correct / total)

        # Test perplexity
        lm = LanguageModel(hmm)
        perplexity = lm.calculatePerplexity(test_sents)

        print(perplexity)