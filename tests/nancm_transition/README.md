# v61 trainable causal expert transition
v59 finite-order corpus integration step passed; v59 native run and v60
corpus run still active at implementation. v59 development promotion
unchanged. Learn bounded contextual corrections to 3x3 transition
logits, seven inputs (bias, observed prefix length, previous space/newline,
capital fraction in previous64 bytes, two clipped log likelihood ratios
from previously completed event). The prior at first event is fixed;
no current target or current likelihood is available to transition.
63 parameters, 96 AdamW updates, exact forward BPTT over padded contiguous
document windows. Fit on primary+secondary unindexed books (previously
used for component credit, not novel); fit grid chooses initial prior/hazard,
calibration chooses scale incl zero, independent v60 switch and static
grid controls. Same initial switch and forced learned control. Preserve
original model/optimizer artifacts, save new transition moments. Finite
order runner only; native option present but not launched. Local executed
test: 81 enumerated paths error 2.7755575615628914e-17, runtime error
5.551115123125783e-17, 63 nonzero gradients max FD1.4628298572461063e-10,
zero scale parity 8.326672684688674e-17. Constructed predictive signal
task 1.1752195429886783 -> .25515018872940326 bits/event; reversed
signal 3.0153105379083427. Not language accuracy. Actual corpus pending.
No inherited geometry gradient yet; learn allocation first. Historical
calibration overlap and prefix-only screening still limitations.
