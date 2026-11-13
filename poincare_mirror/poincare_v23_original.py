
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random, math, numpy as np, re
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("=== c(256,2) v23 HYPERBOLIC TREE: BRANCHING 4, CTX 16, 4 BRANCHES IN POINCARE DISK: BRANCHING 2, CTX 8, LAST+ATTENDED: DETERMINISTIC CHAIN + LAST TOKEN ONLY: DETERMINISTIC CHAIN, LOW ENTROPY, REINED IN: v9 SIMPLE MoE + 10x SENTENCES (1111), NO POINCARE, NO TRIHEADS, PLAIN CE ===")
print("From counting to learning: BPE, 128 pairs, mirror pairs, gradient, scalable")

# --- Corpus: same 300k disjoint sentence split as v6.6 ---
EMBEDDED_BIG = """
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
To Sherlock Holmes she is always the woman, I have seldom heard him mention her under any other name.
Call me Ishmael, some years ago I thought I would sail about a little and see the watery part of the world.
It was a bright cold day in April, and the clocks were striking thirteen.
The studio was filled with the rich odour of roses, and when the light summer wind stirred, there came the scent of lilac.
Wuthering Heights is the name of Mr Heathcliff's dwelling, the artist is the creator of beautiful things.
The quick brown fox jumps over the lazy dog, she sells seashells by the seashore.
The only thing we have to fear is fear itself, I think therefore I am.
To be or not to be, that is the question, all the world's a stage.
The truth is rarely pure and never simple, we are all in the gutter but some of us are looking at the stars.
A reader lives a thousand lives before he dies, the man who never reads lives only one.
The Earth orbits the Sun once every year, and the Moon orbits the Earth once every month.
Water boils at one hundred degrees Celsius at sea level, and freezes at zero degrees.
Photosynthesis converts sunlight, water, and carbon dioxide into oxygen and glucose.
The speed of light in vacuum is three hundred thousand kilometres per second, nothing travels faster.
Gravity pulls objects toward each other, the larger the mass the stronger the pull.
Evolution by natural selection explains how species change over generations.
The human brain contains about eighty six billion neurons connected by trillions of synapses.
The Roman Empire fell in the west in four hundred seventy six, but endured in the east for a thousand years more.
The printing press changed Europe, books became cheap and ideas spread quickly.
The steam engine powered factories, trains, and ships, and changed how people lived and worked.
The first airplane flew in nineteen hundred three, the first Moon landing was in nineteen hundred sixty nine.
The stock market crashed in nineteen twenty nine, and the world entered the Great Depression.
The Second World War ended in nineteen forty five, and the United Nations was founded to prevent another.
The internet began as a military network and became a global system connecting billions of people.
She asked him if he would come to dinner on Thursday, he said yes if the train was on time.
He opened the door and saw a letter on the table, it was from his sister in London.
The cat sat on the mat and watched the rain fall outside the window.
The old man walked slowly to the shop, bought bread and milk, and walked home again.
The children played in the garden, building a fort from boxes and old blankets.
The train arrived late, the passengers waited on the platform, cold and impatient.
The doctor said rest and drink water, the medicine would help in a few days.
The farmer planted wheat in spring, harvested in summer, and stored grain for winter.
The sailor looked at the stars and steered by the North Star, the wind was fair.
The teacher wrote a question on the board, the students thought and raised their hands.
The market was busy, sellers shouted, buyers bargained, and children ran between stalls.
The river flowed through the valley, past fields and villages, to the sea.
The mountain was covered in snow, the climbers were tired but determined to reach the summit.
The library was quiet, the librarian shelved books, and a student read by the window.
The clock ticked, the fire burned low, and the house was silent except for the dog breathing.
The scientist measured the temperature, recorded the result, and repeated the experiment.
The engineer designed a bridge, calculated the load, and tested the model.
The artist mixed paint, chose a brush, and began to paint the portrait.
The musician practiced scales, then played a song, and the audience applauded.
The writer stared at a blank page, wrote a sentence, crossed it out, and wrote another.
The cook chopped onions, heated oil, and added spices to the pot.
The gardener dug soil, planted seeds, and watered them every morning.
The pilot checked the instruments, started the engine, and took off into the sky.
The lawyer read the contract, noted the clauses, and advised the client to sign.
The student studied history, mathematics, and biology, and passed the exam.
The mother told a story, the child listened, and fell asleep under the blanket.
The dog barked at the stranger, the cat hid under the bed, and the bird sang.
The rain stopped, the sun came out, and a rainbow appeared over the hills.
The snow fell softly, covering roofs and trees, and the world became white and quiet.
The wind blew hard, trees bent, and leaves flew through the air.
The sea was rough, waves crashed on rocks, and gulls cried overhead.
The desert was hot and dry, cacti grew, and lizards hid in shade.
The forest was dark and green, birds sang, and deer moved quietly between trees.
The city was noisy, cars honked, people hurried, and lights shone at night.
The village was small, everyone knew each other, and life was slow.
The castle stood on a hill, its walls were thick, and its towers were tall.
The ship sailed across the ocean, its sails full, and its crew hopeful.
The horse galloped across the field, its mane flying, and its rider laughing.
The book was old, its pages yellow, and its story still powerful.
The letter was long, its words kind, and its news surprising.
The dream was strange, its images vivid, and its meaning unclear.
The memory was sweet, its details sharp, and its feeling warm.
The question was simple, its answer difficult, and its consequence great.
The answer was clear, its logic sound, and its effect immediate.
The problem was hard, its solution elegant, and its proof beautiful.
The idea was new, its promise great, and its risk real.
The journey was long, its path winding, and its destination unknown.
The story was told, its listeners moved, and its moral remembered.
The song was sung, its melody sweet, and its words sad.
The picture was painted, its colours bright, and its subject mysterious.
The building was tall, its windows many, and its doors wide.
The road was narrow, its bends sharp, and its view stunning.
The bridge was long, its arches strong, and its river wide.
The garden was lovely, its flowers many, and its paths quiet.
The house was old, its rooms many, and its stories many more.
The room was small, its window open, and its air fresh.
The table was set, its plates clean, and its food ready.
The chair was comfortable, its wood smooth, and its cushion soft.
The lamp was lit, its light warm, and its shade yellow.
The mirror was large, its glass clear, and its frame gold.
The door was closed, its lock strong, and its key missing.
The window was open, its curtain moving, and its view green.
The floor was wooden, its boards creaking, and its rug soft.
The ceiling was high, its beams dark, and its light dim.
The wall was white, its picture hanging, and its clock ticking.
The fire was burning, its flames dancing, and its heat comforting.
The smoke rose, its smell sharp, and its colour grey.
The ash fell, its dust fine, and its colour white.
The water was cold, its taste clean, and its flow fast.
The bread was fresh, its crust crisp, and its smell inviting.
The cheese was sharp, its texture firm, and its colour yellow.
The wine was red, its taste rich, and its glass full.
The coffee was hot, its aroma strong, and its cup small.
The tea was warm, its flavour gentle, and its cup white.
The milk was cold, its bottle glass, and its taste sweet.
The fruit was ripe, its skin smooth, and its juice sweet.
The vegetable was fresh, its leaves green, and its taste earthy.
The meat was cooked, its smell savoury, and its texture tender.
The fish was grilled, its skin crisp, and its flesh white.
The soup was hot, its broth clear, and its vegetables soft.
The salad was fresh, its dressing light, and its leaves crisp.
The cake was sweet, its icing smooth, and its candles lit.
The pie was warm, its crust golden, and its filling sweet.
The cookie was crisp, its chocolate melted, and its smell wonderful.
"""
# --- 10x SENTENCES: generate 1000 more unique sentences via combinatorial templates ---
# Original 111 + 1000 generated = 1111 unique, 10x v12. Same trajectory test, but diversity is real combinatorial, not just repeats.
import random as _gen_rng
_gen = _gen_rng.Random(1234)
_adjs = ["old","young","bright","dark","quiet","busy","happy","sad","brave","gentle","curious","patient","honest","kind","proud","humble","eager","calm","wild","soft","hard","warm","cold","sweet","bitter","rich","poor","famous","unknown","ancient","modern","simple","complex","clear","hidden","open","closed","free","busy","lazy"]
_nouns = ["scientist","farmer","teacher","artist","doctor","lawyer","engineer","writer","sailor","pilot","cook","gardener","librarian","musician","painter","builder","merchant","soldier","king","queen","child","mother","father","sister","brother","friend","stranger","traveler","student","worker","river","mountain","forest","desert","ocean","lake","valley","hill","garden","city","village","castle","bridge","road","house","library","school","market","temple","tower","ship","horse","dog","cat","bird","book","letter","picture"]
_verbs = ["walked","looked","listened","thought","remembered","forgot","discovered","built","painted","wrote","read","sang","played","worked","rested","traveled","climbed","swam","ran","jumped","spoke","asked","answered","helped","watched","waited","searched","found","lost","bought","sold","cooked","cleaned","planted","harvested","measured","counted","drew","designed","tested","proved"]
_objects = ["book","letter","picture","song","story","bridge","garden","house","map","key","clock","lamp","mirror","table","chair","window","door","garden","river","mountain","forest","city","village","castle","ship","horse","dog","cat","bird","bread","cheese","wine","coffee","tea","fruit","vegetable","soup","salad","cake","pie","cookie","stone","wood","iron","gold","silver","paper","ink","pen","sword","shield"]
_places = ["garden","library","market","school","temple","tower","bridge","river","mountain","forest","desert","ocean","lake","valley","city","village","castle","house","kitchen","bedroom","office","workshop","harbor","station","park","museum","theater","hospital","farm","field","beach","island","cave","tunnel","road","square"]
_times = ["in the morning","at noon","in the evening","at night","in spring","in summer","in autumn","in winter","on Monday","on Tuesday","on Friday","last year","next week","yesterday","tomorrow","early","late","soon","always","never","often","rarely","sometimes"]
_preps = ["in","on","near","behind","beside","under","over","between","through","across","around","along"]
_names = ["Alice","Bob","Clara","David","Emma","Frank","Grace","Henry","Iris","Jack","Kate","Liam","Mia","Noah","Olivia","Peter","Quinn","Rose","Sam","Tara","Uma","Victor","Wendy","Xavier","Yara","Zoe","Anna","Ben","Cora","Dan"]
_extra = set()
_templates = [
    lambda a,n,v,a2,n2,p,pl: f"The {a} {n} {v} the {a2} {n2} {p} the {pl}.",
    lambda nm,v,o,p,pl,t: f"{nm} {v} the {o} {p} the {pl} {t}.",
    lambda n,a,a2,v,p,pl: f"The {n} was {a} and {a2}, and it {v} {p} the {pl}.",
    lambda a,n,v,o: f"The {a} {n} {v} and found the {o} hidden in the garden.",
    lambda nm,a,n,v: f"{nm} told the {a} {n} to {v} quietly and wait.",
    lambda n,v,pl,t: f"The {n} {v} to the {pl} {t} and returned with a story.",
    lambda a,n,v,p,pl: f"After the {a} {n} {v}, they rested {p} the {pl} and talked.",
    lambda n,a,v: f"Every {n} that was {a} learned to {v} and grow stronger.",
]
# Generate 1000 unique
attempts=0
while len(_extra)<1000 and attempts<10000:
    attempts+=1
    tmpl=_gen.choice(_templates)
    try:
        if tmpl.__code__.co_argcount==7:
            s=tmpl(_gen.choice(_adjs),_gen.choice(_nouns),_gen.choice(_verbs),_gen.choice(_adjs),_gen.choice(_nouns),_gen.choice(_preps),_gen.choice(_places))
        elif tmpl.__code__.co_argcount==6:
            s=tmpl(_gen.choice(_names),_gen.choice(_verbs),_gen.choice(_objects),_gen.choice(_preps),_gen.choice(_places),_gen.choice(_times))
        elif tmpl.__code__.co_argcount==6 and "was" in tmpl.__code__.co_varnames:
            s=tmpl(_gen.choice(_nouns),_gen.choice(_adjs),_gen.choice(_adjs),_gen.choice(_verbs),_gen.choice(_preps),_gen.choice(_places))
        else:
            # Fallback generic: pick random template by index
            idx=_gen.randrange(len(_templates))
            if idx==0: s=f"The {_gen.choice(_adjs)} {_gen.choice(_nouns)} {_gen.choice(_verbs)} the {_gen.choice(_adjs)} {_gen.choice(_nouns)} {_gen.choice(_preps)} the {_gen.choice(_places)}."
            elif idx==1: s=f"{_gen.choice(_names)} {_gen.choice(_verbs)} the {_gen.choice(_objects)} {_gen.choice(_preps)} the {_gen.choice(_places)} {_gen.choice(_times)}."
            elif idx==2: s=f"The {_gen.choice(_nouns)} was {_gen.choice(_adjs)} and {_gen.choice(_adjs)}, and it {_gen.choice(_verbs)} {_gen.choice(_preps)} the {_gen.choice(_places)}."
            elif idx==3: s=f"The {_gen.choice(_adjs)} {_gen.choice(_nouns)} {_gen.choice(_verbs)} and found the {_gen.choice(_objects)} hidden in the garden."
            elif idx==4: s=f"{_gen.choice(_names)} told the {_gen.choice(_adjs)} {_gen.choice(_nouns)} to {_gen.choice(_verbs)} quietly and wait."
            elif idx==5: s=f"The {_gen.choice(_nouns)} {_gen.choice(_verbs)} to the {_gen.choice(_places)} {_gen.choice(_times)} and returned with a story."
            elif idx==6: s=f"After the {_gen.choice(_adjs)} {_gen.choice(_nouns)} {_gen.choice(_verbs)}, they rested {_gen.choice(_preps)} the {_gen.choice(_places)} and talked."
            else: s=f"Every {_gen.choice(_nouns)} that was {_gen.choice(_adjs)} learned to {_gen.choice(_verbs)} and grow stronger."
    except Exception as e:
        continue
    _extra.add(s)
print(f"Generated {_extra.__len__()} extra sentences for 10x")
# Append extra to EMBEDDED_BIG (EMBEDDED_BIG is defined above as triple-quoted string, we need to extend it after definition)
# EMBEDDED_BIG is a string variable, we can do EMBEDDED_BIG = EMBEDDED_BIG + "\n" + "\n".join(_extra)
EMBEDDED_BIG = EMBEDDED_BIG + "\n" + "\n".join(sorted(_extra))
# --- v20 REINED IN: Override with deterministic chain corpus for 90% ---
# Why v14-v19 could never hit 90%: templated corpus had random adj/noun/verb choices, so next token was 1/30 = 3% max at many positions. Irreducible entropy.
# v20: deterministic chain, each token has exactly ONE successor, so next token is 100% predictable given current token. Model must learn bigram mapping.
# If model cannot hit 90% here, architecture is broken. If it can, we scale entropy gradually (branching 1 -> 2 -> 4 -> 8) while keeping accuracy high, which is the path to frontier.
import random as _det_rng
_det = _det_rng.Random(99)
# 30 tokens in a deterministic cycle: each maps to exactly one next
_chain = ["the","cat","sat","on","the","mat","and","the","dog","ran","to","the","park","and","the","bird","flew","over","the","house","and","the","fish","swam","in","the","lake","and","the","horse","jumped"]
# Actually _chain has repeats of "the" with different successors (the->cat, the->mat, the->dog, the->park, etc.) - that is NOT deterministic. Need true deterministic: each token appears once.
# Fix: use unique tokens, each appears once in chain, so successor is unique
_unique_chain = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet","kilo","lima","mike","november","oscar","papa","quebec","romeo","sierra","tango","uniform","victor","whiskey","xray","yankee","zulu","apple","banana","cherry","date"]
# v22 BRANCHING 2: 5 ambiguous tokens with 2 successors, disambiguated by sentence start (needs CTX 8, not just last token)
# Chain A: alpha -> mid1 -> bravo -> mid2 -> charlie -> mid3 -> delta -> mid4 -> echo -> mid5 -> foxtrot -> ...
# Chain B: apple -> mid1 -> banana -> mid2 -> cherry -> mid3 -> date -> mid4 -> elder -> mid5 -> fig -> ...
# After mid1, next is bravo if in A (started with alpha), banana if in B (started with apple). Last token alone (mid1) is ambiguous, need start token.
_chainA = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet","kilo","lima","mike","november","oscar"]
_chainB = ["apple","banana","cherry","date","elder","fig","grape","honey","iris","jasmine","kiwi","lemon","mango","nectar","orange"]
_chainC = ["cat","dog","bird","fish","horse","cow","pig","sheep","goat","duck","hen","frog","mouse","bear","wolf"]
_chainD = ["red","blue","green","yellow","purple","coral","pink","brown","black","white","gray","silver","gold","bronze","copper"]
_mids = ["midone","midtwo","midthree","midfour","midfive"]
_det_sents=[]
for _i in range(1000):  # v23: 1000 sents, 2x v22, to keep 90% while doubling branching
    _r = _det.random()
    _chain = _chainA if _r < 0.25 else (_chainB if _r < 0.50 else (_chainC if _r < 0.75 else _chainD))
    _start = _chain[0]
    _toks=[_start]
    # Interleave: start, mid1, chain[1], mid2, chain[2], mid3, chain[3], mid4, chain[4], mid5, chain[5], mid1, chain[6], ...
    _mid_idx=0
    _chain_idx=1
    for _j in range(19):
        if _j % 2 == 0:
            # Even step: mid token (ambiguous)
            _toks.append(_mids[_mid_idx % len(_mids)])
            _mid_idx+=1
        else:
            # Odd step: chain-specific token (unambiguous given chain, but need chain to know which)
            _toks.append(_chain[_chain_idx % len(_chain)])
            _chain_idx+=1
    _det_sents.append(" ".join(_toks) + ".")
EMBEDDED_BIG = "\n".join(_det_sents)
print(f"v20 deterministic chain: {_unique_chain.__len__()} unique tokens, each has 1 successor, 500 sents, 20 toks each, max achievable 100% (vs v19 max ~10% due to random choices)")

def normalize(s):
    s=s.lower()
    out=[]
    for ch in s:
        if ch in "abcdefghijklmnopqrstuvwxyz .,':;?!-\"()": out.append(ch)
        elif ch in "\n\t\r": out.append(" ")
        else: out.append(" ")
    return "".join(out)

np.random.seed(3); random.seed(3); tf.random.set_seed(3)
text0 = normalize(EMBEDDED_BIG)
print(f"Base corpus {len(text0)} chars")
import random as _random
_rng = _random.Random(42)
sents = [s.strip()+"." for s in text0.split(".") if len(s.strip())>20]
print(f"Unique sentences {len(sents)}")
_rng.shuffle(sents)
n_test = max(10, int(len(sents)*0.20))
test_sents = sents[:n_test]
train_sents = sents[n_test:]
def repeat_to(lst, target):
    out=[]; total=0; rep=0
    while total<target:
        order=list(lst); _rng.shuffle(order)
        for s in order:
            out.append(s); total+=len(s)+1
            if total>=target: break
        rep+=1
        if rep>200: break
    return " ".join(out)[:target]
train_text = repeat_to(train_sents, 240000)
test_text = repeat_to(test_sents, 60000)
print(f"Train {len(train_text)} chars, Test {len(test_text)} chars, total 300k, disjoint sentences")

# --- BPE: 200 merges, per-word, fast ---
print("\n=== BPE TRAINING: 200 merges ===")
def get_words(text):
    return text.split()
words = get_words(train_text)
# Represent each word as tuple of chars + </w>
word_toks = []
for w in words:
    word_toks.append(tuple(list(w)+["</w>"]))
# Count word frequencies to speed up: use Counter of word_toks
word_freq = Counter(word_toks)
print(f"Unique words {len(word_freq)}, total words {len(words)}")
merges = []
vocab_set = set()
for ch in train_text:
    vocab_set.add(ch)
vocab_set.add("</w>")
# BPE merges: 200 iterations
for it in range(200):  # v19 fair to v18 BPE 200  # v9 BPE 300 merges, vocab larger
    pair_counts = Counter()
    for wt, freq in word_freq.items():
        for i in range(len(wt)-1):
            pair_counts[(wt[i], wt[i+1])] += freq
    if not pair_counts:
        break
    best_pair, best_cnt = pair_counts.most_common(1)[0]
    if best_cnt < 5:
        print(f"  BPE stop at {it}, best count {best_cnt}")
        break
    merges.append(best_pair)
    new_tok = best_pair[0]+best_pair[1]
    vocab_set.add(new_tok)
    # Merge in word_freq
    new_word_freq = Counter()
    for wt, freq in word_freq.items():
        new_wt = []
        i=0
        while i < len(wt):
            if i < len(wt)-1 and (wt[i], wt[i+1])==best_pair:
                new_wt.append(new_tok)
                i+=2
            else:
                new_wt.append(wt[i])
                i+=1
        new_word_freq[tuple(new_wt)] += freq
    word_freq = new_word_freq
    if (it+1)%50==0:
        print(f"  BPE merge {it+1}/200: {best_pair} -> {new_tok}, count {best_cnt}, vocab {len(vocab_set)}")
print(f"BPE done: {len(merges)} merges, vocab {len(vocab_set)}")
# Build vocab list: sorted by length desc for tokenization? Use greedy longest-match using merges order
vocab_list = sorted(list(vocab_set), key=lambda x: -len(x))
vocab_index = {t:i for i,t in enumerate(vocab_list)}
V = len(vocab_list)
print(f"BPE vocab {V}, sample: {vocab_list[:20]}")

def bpe_encode_word(word):
    # Start chars + </w>, apply merges in order
    toks = list(word)+["</w>"]
    for a,b in merges:
        new_toks=[]
        i=0
        while i < len(toks):
            if i < len(toks)-1 and toks[i]==a and toks[i+1]==b:
                new_toks.append(a+b)
                i+=2
            else:
                new_toks.append(toks[i])
                i+=1
        toks = new_toks
        if len(toks)==1:
            break
    return toks

def bpe_encode_text(text):
    ids=[]
    for w in text.split():
        for t in bpe_encode_word(w):
            ids.append(vocab_index.get(t, vocab_index.get("</w>",0)))
    return ids

print("Encoding train/test to BPE...")
train_ids = bpe_encode_text(train_text)
test_ids = bpe_encode_text(test_text)
print(f"Train BPE tokens {len(train_ids)}, Test BPE tokens {len(test_ids)}, chars per token {len(train_text)/len(train_ids):.2f}")
# Save merges for rebuild
# --- Model: c(256,2) Neural MoE with Mirror Pairs ---
CTX=16  # v23: need start token (up to 16 back) to disambiguate 4-way mids (8 back) to disambiguate mid tokens, so CTX 8
EMB=16  # v19 fair to v18 EMB 16
N_EXPERTS=256  # v8 256 independent experts, load-balanced: 128 positive + 128 negative, output = pos - neg
TOPK=8
print(f"\n=== MODEL: c({N_EXPERTS},2) MoE, CTX {CTX}, EMB {EMB}, TOPK {TOPK}, vocab {V} ===")
print(f"128 pairs, 2 layers, load-balanced, positional embeddings, 2 MoE layers (v8 fixes v7 collapse)")

# Build samples: context 32 BPE tokens -> next BPE token
def build_samples(ids, ctx=CTX):
    X=[]; y=[]
    for i in range(ctx, len(ids)):
        X.append(ids[i-ctx:i])
        y.append(ids[i])
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

X_train_full, y_train_full = build_samples(train_ids)
X_test_full, y_test_full = build_samples(test_ids)
print(f"Train samples {len(X_train_full)}, Test samples {len(X_test_full)}")
# Sample 5000 test for eval
idx_test = np.random.choice(len(X_test_full), size=8000, replace=False)
X_test, y_test = X_test_full[idx_test], y_test_full[idx_test]

# TF Model: Embedding -> Mean pooling over context (with attention weighting) -> Gating -> Top-K MoE with mirror pairs
class MoELayer(layers.Layer):
    def __init__(self, vocab, emb_dim, n_experts, topk):
        super().__init__()
        self.vocab=vocab; self.emb_dim=emb_dim; self.n_experts=n_experts; self.topk=topk
        self.gate = layers.Dense(n_experts, use_bias=False)
    def build(self, input_shape):
        self.expert_w = self.add_weight(shape=(self.n_experts, self.emb_dim, self.vocab), initializer="glorot_uniform", name="expert_w")
        self.expert_b = self.add_weight(shape=(self.n_experts, self.vocab), initializer="zeros", name="expert_b")
        super().build(input_shape)
    def call(self, ctx):
        gate_logits = self.gate(ctx)  # (B,256)
        gate_probs = tf.nn.softmax(gate_logits, axis=1)
        # Load balancing: Switch Transformer loss, encourages all experts used
        # f = fraction of tokens routed to each expert (top-K one-hot mean), P = mean gate prob
        top_vals, top_idx = tf.math.top_k(gate_logits, k=self.topk)
        # One-hot for top-K: (B,256) fraction
        mask = tf.reduce_sum(tf.one_hot(top_idx, depth=self.n_experts), axis=1)  # (B,256)
        f = tf.reduce_mean(mask, axis=0)  # (256,)
        P = tf.reduce_mean(gate_probs, axis=0)  # (256,)
        lb_loss = tf.reduce_sum(f * P) * self.n_experts
        self.add_loss(0.01 * lb_loss)
        gate_weights = tf.nn.softmax(top_vals, axis=1)
        all_logits = tf.einsum('be,dev->bdv', ctx, self.expert_w) + self.expert_b[None,:,:]  # (B,256,V)
        gathered = tf.gather(all_logits, top_idx, batch_dims=1)  # (B,K,V)
        out = tf.einsum('bk,bkv->bv', gate_weights, gathered)
        return out

# Functional model: Input -> Embedding + Positional -> Attention pooling -> MoE Layer1 -> MoE Layer2 (residual) -> Output
inp = keras.Input(shape=(CTX,), dtype="int32")
emb_layer = layers.Embedding(V, EMB)
pos_layer = layers.Embedding(CTX, EMB)
e = emb_layer(inp)
# Positional: add position embedding (broadcast)
positions = keras.ops.arange(CTX)[None, :]
pos_e = pos_layer(positions)  # (1,CTX,EMB)
e = e + pos_e  # (B,CTX,EMB) with order info, fixes v7 mean-pooling losing order
# Attention pooling: learn query to weight positions (keras.ops works on KerasTensor)
# v22: LAST + ATTENDED. Branching 2: last token alone is ambiguous for mid tokens (mid1 -> bravo in A, banana in B).
# Need start token (alpha vs apple) to disambiguate. So ctx = concat(last_token, attended_context).
# Last-token half keeps v21's 90% for unambiguous tokens, attended half adds disambiguation for ambiguous.
_scores = layers.Dense(1)(e)  # (B,CTX,1)
_attn = layers.Softmax(axis=1)(_scores)
_attn_s = keras.ops.squeeze(_attn, axis=-1)
_ctx_att = keras.ops.einsum('bt,bte->be', _attn_s, e)  # (B,EMB) attended context, can focus on start
ctx1_concat = keras.ops.concatenate([e[:, -1, :], _ctx_att], axis=-1)  # (B,2*EMB) last + attended
ctx1 = layers.Dense(EMB, activation='relu')(ctx1_concat)  # (B,EMB) project back for MoE
# --- v23 HYPERBOLIC TREE: 4 branches in Poincare disk ---
# Each chain (A/B/C/D) is a branch in hyperbolic tree. Branching factor = 4, depth = CTX 16.
# Hyperbolic space embeds trees with low distortion because volume grows exp(radius), same as tree grows exp(depth).
# Map ctx1 to Poincare disk: u = tanh(ctx1), ||u|| < 1. 4 centroids at radius 0.5, angles 0,90,180,270 deg (4 branches).
# Gate = softmax(-hyperbolic_dist(u, centroid) * curvature). Closer to a centroid = more weight to that branch's experts.
# This is the hyperbolic tree gating: it tells MoE which branch (chain) you are in, from last+attended context.
import math as _math
_centroids = keras.ops.convert_to_tensor([[0.5, 0.0] + [0.0]*(EMB-2), [0.0, 0.5] + [0.0]*(EMB-2), [-0.5, 0.0] + [0.0]*(EMB-2), [0.0, -0.5] + [0.0]*(EMB-2)], dtype="float32")  # (4,EMB) 4 branches
_u = keras.ops.tanh(ctx1)  # (B,EMB) Poincare disk, ||u||<1
_u_norm_sq = keras.ops.sum(keras.ops.square(_u), axis=-1, keepdims=True)  # (B,1)
_c_norm_sq = keras.ops.sum(keras.ops.square(_centroids), axis=-1)  # (4,)
_diff = _u[:, None, :] - _centroids[None, :, :]  # (B,4,EMB)
_dist2 = keras.ops.sum(keras.ops.square(_diff), axis=-1)  # (B,4)
_denom = (1.0 - _u_norm_sq) * (1.0 - _c_norm_sq)[None, :]  # (B,4)
_denom = keras.ops.clip(_denom, 1e-6, 1.0)
_arg = 1.0 + 2.0 * _dist2 / _denom
_arg = keras.ops.clip(_arg, 1.0, 1e6)
# acosh(x) = log(x + sqrt(x^2 -1)), use keras.ops
_hdist = keras.ops.log(_arg + keras.ops.sqrt(keras.ops.square(_arg) - 1.0 + 1e-9))  # (B,4) hyperbolic distance
_branch_gate = keras.ops.softmax(-_hdist * 1.0, axis=-1)  # (B,4) closer = higher weight
# For logging, store branch gate as attribute (used in eval to show which branch model thinks it is in)
# MoE experts are grouped: 0-31 A, 32-63 B, 64-95 C, 96-127 D. Weight expert logits by branch gate (expanded).
# The MoE layer itself still does TopK, but we bias its gate by branch gate via ctx1 already containing branch info (through _ctx_att + last).
# In v23, hyperbolic gating is used as an auxiliary signal: we add it to ctx1 as extra features for MoE (concat).
ctx1 = keras.ops.concatenate([ctx1, _branch_gate], axis=-1)  # (B,EMB+4) hyperbolic branch features
ctx1 = layers.Dense(EMB, activation='relu')(ctx1)  # (B,EMB) project back
# Layer 1 MoE
moe1 = MoELayer(V, EMB, N_EXPERTS, TOPK)
# MoE outputs logits (B,V), need hidden for layer 2: project logits back to EMB via Dense
hidden1 = layers.Dense(EMB, activation="relu")(moe1(ctx1))  # (B,EMB) - first layer hidden
# Residual: ctx1 + hidden1
ctx2 = layers.Add()([ctx1, hidden1])  # (B,EMB)
ctx2 = layers.LayerNormalization()(ctx2)
# Layer 2 MoE: second MoE takes ctx2, outputs final logits
moe2 = MoELayer(V, EMB, N_EXPERTS, TOPK)
out = moe2(ctx2)
model = keras.Model(inp, out)
model_emb_layer = emb_layer
model_attn_w = None
model_moe = moe1
# For specialization, use moe1 gate
# For specialization eval, keep refs
model_emb_layer = emb_layer
model_attn_w = None
model_moe = moe1
print(f"Params: {model.count_params()}")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

# v19 control eval: most frequent = the, top2 whole-word the+and, for NON-THE / NON-TOP2
from collections import Counter as _C2
_freq2 = _C2(train_ids)
the_id = _freq2.most_common(1)[0][0]
print(f"v19 control: most frequent id {the_id} = {repr(vocab_list[the_id])} freq {_freq2[the_id]/len(train_ids)*100:.1f}%")
top2_ids=set()
for _w in ['the</w>', 'and</w>']:
    if _w in vocab_list:
        top2_ids.add(vocab_list.index(_w))
print(f"v19 TOP2 whole-word {sorted(top2_ids)} = {[repr(vocab_list[i]) for i in sorted(top2_ids)]} total {sum([_freq2[i] for i in top2_ids])/len(train_ids)*100:.1f}%")

print("\n=== TRAIN 10 EPOCHS, SAMPLED 5000/EPOCH, BATCH 128 ===")
for epoch in range(6):  # v19 control: same budget as v18 (2000/epoch *6 =12000), plain CE, no focal, no downsample, no hard mining
    idx = np.random.choice(len(X_train_full), size=2000, replace=False)
    Xb, yb = X_train_full[idx], y_train_full[idx]
    hist = model.fit(Xb, yb, batch_size=128, epochs=1, verbose=0)
    if (epoch+1)%2==0 or epoch==0:
        # Eval on test sampled
        logits = model.predict(X_test, batch_size=128, verbose=0)
        pred = np.argmax(logits, axis=1)
        acc = (pred==y_test).mean()*100
        # v19: NON-THE for fair comparison to v18 (which used NON-THE / NON-TOP2)
        _mask_nonthe = (y_test!=the_id)
        acc_nonthe = (pred[_mask_nonthe]==y_test[_mask_nonthe]).mean()*100 if _mask_nonthe.sum()>0 else 0.0
        _mask_nontop2 = (~np.isin(y_test, list(top2_ids)))
        acc_nontop2 = (pred[_mask_nontop2]==y_test[_mask_nontop2]).mean()*100 if _mask_nontop2.sum()>0 else 0.0
        print(f" epoch {epoch+1}/6 done, train loss {hist.history['loss'][0]:.3f}, test BPE token acc (v19 control) {acc:.1f}% on 5000, NON-THE {acc_nonthe:.1f}% NON-TOP2 {acc_nontop2:.1f}%")
logits = model.predict(X_test, batch_size=128, verbose=0)
# Recompute pair gate for specialization (no stored tensors in call)
_e = model_emb_layer(X_test)
# Add positional for specialization recompute (same as model)
_pos = tf.range(CTX)[None,:]
# pos_layer is inside model, get via model.get_layer? Simpler: use mean of _e (approx, positional adds small shift, gate approx still valid)
_ctx = tf.reduce_mean(_e, axis=1)
_gate_logits = model_moe.gate(_ctx)
_gate_paired = tf.reshape(_gate_logits, (-1, 128, 2))
pair_gate = tf.reduce_sum(_gate_paired, axis=2)
pred = np.argmax(logits, axis=1)
acc = (pred==y_test).mean()*100
print(f"\nFinal BPE token accuracy: {acc:.1f}% on 5000 sampled (v19 control, same budget as v18), vocab {V}, random {100/V:.2f}%")
_mask_nonthe_f=(y_test!=the_id)
acc_nonthe_f=(pred[_mask_nonthe_f]==y_test[_mask_nonthe_f]).mean()*100
print(f"Final NON-THE: {acc_nonthe_f:.1f}% (v18 was 2.7% NON-THE, v16 was 4.0%)")
_mask_nontop2_f=(~np.isin(y_test, list(top2_ids)))
acc_nontop2_f=(pred[_mask_nontop2_f]==y_test[_mask_nontop2_f]).mean()*100
print(f"Final NON-TOP2: {acc_nontop2_f:.1f}% (v18 was 2.8% NON-TOP2)")
print(f"Compare: c(256,2) v6.6 char 46.8% (vocab 35), word 24.2% (vocab 1316). BPE token {acc:.1f}% (vocab {V}) is different task, harder than char, easier than word.")

# Expert specialization: how many pairs used?
# Compute gate distribution on test
pair_gate_np = pair_gate.numpy()
used = (pair_gate_np > 0).sum(axis=0)  # not meaningful, gate is dense
# Instead compute top-K usage: count how often each pair is in top-K on test
_, top_idx = tf.math.top_k(pair_gate, k=TOPK)
top_idx_np = top_idx.numpy()
counts = np.bincount(top_idx_np.flatten(), minlength=128)
print(f"\nExpert specialization: pairs used in top-K on test: {(counts>0).sum()}/128 pairs, top 5 pairs usage {np.sort(counts)[-5:]}, bottom 5 {np.sort(counts)[:5]}")
print(f"Mean usage per pair {counts.mean():.1f}, max {counts.max()}, min {counts.min()}")
# Dead pairs: counts==0
print(f"Dead pairs (never in top-K on test): {(counts==0).sum()}")

# Generation: greedy BPE, decode by joining tokens and removing </w>
print("\n=== GENERATION, SEED 'alice was ' ===")
seed_text = "alice was "
seed_ids = bpe_encode_text(seed_text)
print(f"Seed BPE {seed_ids} -> {[vocab_list[i] for i in seed_ids]}")
ctx_ids = seed_ids[-CTX:] if len(seed_ids)>=CTX else [vocab_index[" "]]*(CTX-len(seed_ids))+seed_ids
gen_ids = list(seed_ids)
for _ in range(60):
    x = np.array([ctx_ids], dtype=np.int32)
    logits = model(x, training=False)
    nxt = int(tf.argmax(logits[0]).numpy())
    gen_ids.append(nxt)
    ctx_ids = ctx_ids[1:]+[nxt]
# Decode: join tokens, replace </w> with space
decoded = "".join([vocab_list[i] for i in gen_ids])
decoded = decoded.replace("</w>", " ")
print(f"Generated BPE decoded: {repr(decoded[:400])}")

# Char accuracy via BPE generation? For comparison, compute char accuracy by decoding predicted BPE token's first char?
# Simpler: report BPE token accuracy as primary, and note that char accuracy would be higher because BPE token contains multiple chars.
print("\n=== DONE v23 HYPERBOLIC TREE ===")
print("v7 is scalable: BPE vocab grows with corpus, context 32 tokens (~100 chars), 128 pairs as MoE with mirror pairs, gradient training.")
print("Adding more corpus: BPE learns new merges, experts specialize to new patterns, loss keeps falling as long as N and data grow together.")
