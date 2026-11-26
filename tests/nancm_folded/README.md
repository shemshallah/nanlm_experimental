# v42 case-folded event diagnostic
One probability law supplies literal raw byte probabilities, prefix-free subword
probabilities and exact case-folded ASCII completed-word event probabilities.
A finite order-3 donor permits forward-state aggregation: at most 8 casing
states per folded prefix. Surface case and nonletter delimiter distributions
fit on reference documents only. The raw donor can incorporate a fixed event
entry hidden state and three local lexical-byte rails; the lexical distribution
comes from the inherited field, memory and expanded reference retrieval.
Calibration selects the mixture coefficients and includes zero lexical/raw-field
weights. All inherited arrays and Adam moments remain unchanged.
Executed local test: 34 explicit surface variants, conditional error
1.1102230246251565e-16, token marginal error 2.7755575615628914e-16,
certified exact folded ranking, zero lexical weight raw-donor parity.
The experiment uses completed v36 to avoid blocked v37-v40 dependencies, not a
replacement model. It does not implement Unicode casefold, the original dynamic
v37 byte head, or the v41 persistent graph. Cap-terminated runs are excluded from
complete-word rank but retain byte probability mass. Ranking certification is
reported, not assumed. Corpus results are pending.
