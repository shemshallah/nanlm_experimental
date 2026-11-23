CXX ?= c++
CPPFLAGS ?= -Isrc -Iparent/src -Iparent/integration/v152_native
CXXFLAGS ?= -O2 -std=c++17 -fPIC
.PHONY: all test clean v160 test-v160
all: v167
# v167 is the active development baseline; prior targets remain recoverable.
bin/libwordfield159.so: src/wordfield159_api.cpp src/wordfield158_embed159.cpp src/wordfield158_api.cpp src/wordfield157_api.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test: all
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v167/test_native167.py

.PHONY: test-inherited
test-inherited: bin/libwordfield159.so
	OPENBLAS_NUM_THREADS=1 python tests/test_pointer158.py
	OPENBLAS_NUM_THREADS=1 python tests/test_native159.py
clean:
	rm -f bin/libwordfield159.so

# v160 is an explicit research target; existing v159 default stays unchanged.
v160: bin/libwordfield160.so
bin/libwordfield160.so: src/wordfield160_api.cpp src/wordfield159_embed160.cpp src/wordfield159_api.cpp src/wordfield158_embed159.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test-v160: v160
	OPENBLAS_NUM_THREADS=1 python tests/test_native160.py

# One unified NaNCM lineage; v158 remains default, v161 opt-in research.
.PHONY: v161 test-v161
v161: bin/libwordfield161.so
bin/libwordfield161.so: src/wordfield161_api.cpp src/wordfield159_embed160.cpp src/wordfield159_api.cpp src/wordfield158_embed159.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test-v161: v161
	OPENBLAS_NUM_THREADS=1 python tests/test_native161.py

# Domain-adaptive v162 remains an opt-in development candidate.
.PHONY: v162 test-v162
v162: bin/libwordfield162.so
bin/libwordfield162.so: src/wordfield162_api.cpp src/wordfield159_embed160.cpp src/wordfield158_embed159.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test-v162: v162
	OPENBLAS_NUM_THREADS=1 python tests/test_native162.py

# v163 experimental learned signed-cloud gate; protects v158 default.
.PHONY: v163 test-v163
v163: bin/libwordfield163.so
bin/libwordfield163.so: src/wordfield163_api.cpp src/wordfield162_api.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test-v163: v163
	OPENBLAS_NUM_THREADS=1 python tests/test_native163.py

# v164 opt-in multi-candidate normalized whole-word distribution.
.PHONY: v164 test-v164
v164: bin/libwordfield164.so
bin/libwordfield164.so: src/wordfield164_api.cpp src/wordfield162_embed163.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test-v164: v164
	OPENBLAS_NUM_THREADS=1 python tests/test_native164.py
	OPENBLAS_NUM_THREADS=1 python tests/test_documentation_guard164.py

.PHONY: v165 test-v165
v165: bin/libwordfield165.so
bin/libwordfield165.so: src/wordfield165_api.cpp src/wordfield164_embed165.cpp src/wordfield164_api.cpp src/wordfield162_embed163.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) $< -o $@
test-v165: v165
	OPENBLAS_NUM_THREADS=1 python tests/test_native165.py

# Experimental signed world-state lexical-birth head, exact v164 opt-out.
.PHONY: v166 test-v166
v166: bin/libwordfield166.so
bin/libwordfield166.so: experiments/nancm/v166/wordfield166_api.cpp src/wordfield164_embed165.cpp src/wordfield162_embed163.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@
test-v166: v166
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_native166.py --quick
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_doc_guard166.py

.PHONY: test-v166-full
test-v166-full: v166
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_native166.py
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_doc_guard166.py

# Active v167 inference path, using immutable v166 model and optimizer lineage.
.PHONY: v167 test-v167 test-v167-full
v167: bin/libwordfield167.so
bin/libwordfield167.so: experiments/nancm/v167/wordfield167_api.cpp src/wordfield164_embed165.cpp src/wordfield162_embed163.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@
test-v167: v167
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v167/test_native167.py
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v167/test_trust167.py
test-v167-full: test-v167
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_native166.py --quick
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v166/test_doc_guard166.py

# Experimental v168: native train-book learned causal gating; v167 default retained.
.PHONY: v168 test-v168
v168: bin/libwordfield168.so
bin/libwordfield168.so: experiments/nancm/v168/wordfield168_api.cpp src/wordfield164_embed165.cpp src/wordfield162_embed163.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@

# Native analogical birth composition, default-disabled to protect v167.
.PHONY: v168-ann
v168-ann: bin/libwordfield168_ann.so
bin/libwordfield168_ann.so: experiments/nancm/v168/wordfield168_ann_api.cpp experiments/nancm/v168/wordfield168_api.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@

.PHONY: test-v168-ann
test-v168-ann: v168-ann
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v168/test_native_ann168.py

# v169 experimental symbiotic signed-neighbor lexical rescoring
.PHONY: v169 test-v169
v169: bin/libwordfield169.so
bin/libwordfield169.so: experiments/nancm/v169/wordfield169_api.cpp experiments/nancm/v169/wordfield168_embed_ann169.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@
test-v169: v169
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v169/test_native169.py

# Experimental v169 native signed lexical/ANN symbiotic byte inference.
.PHONY: v169-byte-pack test-v169-byte test-v169-full
v169-byte-pack: checkpoints/v169_native_byte_lattice.bin
checkpoints/v169_native_byte_lattice.bin: checkpoints/v169_byte_ngram.json checkpoints/v169_byte_bridge.json experiments/nancm/v169/pack_byte169.py
	python experiments/nancm/v169/pack_byte169.py
test-v169-byte: v169 v169-byte-pack
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v169/test_native_byte169.py
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v169/test_streaming169.py
test-v169-full: test-v169 test-v169-byte
# Append to the complete v169 recovery Makefile (opt-in development).
.PHONY: v170 test-v170
v170: bin/libwordfield170.so
bin/libwordfield170.so: experiments/nancm/v170/wordfield170_api.cpp experiments/nancm/v170/native_online170.cpp experiments/nancm/v169/wordfield169_api.cpp experiments/nancm/v169/native_byte169.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@
test-v170: v170 v169-byte-pack
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v170/test_native170.py
# NaNCM v171 opt-in signed ANN candidate scaling, no new trained weights.
include experiments/nancm/v171/Makefile.fragment

# v172 opt-in causal signed multirail and trained whole-word teacher.
include experiments/nancm/v172/Makefile.fragment

# v173 trained exact CE + causal normalized byte/word first-letter joint, default off.
include experiments/nancm/v173/Makefile.fragment

# Opt-in dual-polarity, 128D causal temporal word state, inherited 256-byte ABI.
include experiments/nancm/v174/Makefile.fragment

# v175 causal 48-word signed episodic pathway
.PHONY: v175 test-v175
v175: bin/libwordfield175.so
bin/libwordfield175.so: experiments/nancm/v175/wordfield175_api.cpp experiments/nancm/v175/native_temporal175.cpp experiments/nancm/v174/wordfield174_api.cpp experiments/nancm/v174/native_temporal174.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -shared $(CPPFLAGS) -Isrc $< -o $@
test-v175: v175
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/test_native175.py --n 50
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/test_byte_rollback175.py
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/eval_book_disjoint175.py --split val
	OPENBLAS_NUM_THREADS=1 python experiments/nancm/v175/eval_book_disjoint175.py --split test

# NaNCM v176 signed word top-five × firstletter × BPE × verified byte third code
include experiments/nancm/v176/Makefile.fragment

# v177 opt-in causal shared-prefix DAG exact speculative multi-position verifier
include experiments/nancm/v177/Makefile.fragment

# v178 clone-free shared verifier extension
include experiments/nancm/v178/Makefile.fragment

# v179 opt-in COW workers
include experiments/nancm/v179/Makefile.fragment
