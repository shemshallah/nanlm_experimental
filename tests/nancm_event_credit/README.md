# v45 unified event credit
Exact likelihood for literal surface events and their folded marginal, using
the v42 Event and v39 Transport law. For capped runs the subset is all words
with the observed prefix; for empty nonletter events the lexical mass is zero.
No token loss is added on top of the same byte likelihood.
Actual executed CPU tests on PyTorch 2.10.0: 7 events x 4 weights, maximum
literal error 1.734723475976807e-18, folded error 1.1102230246251565e-16,
token marginal error 7.771561172376096e-16, 21 lift-gradient comparisons
maximum 1.7994349066352555e-11. Empty and zero-weight gradients exactly zero.
Actual constructed integrated test (384 train, 192 validation, 192 updates):
raw BPB before 1.5931306168415726; folded-only 1.0735665869783024; literal
1.0737055158009776; literal with wrong history 1.9390137389796394.
Both trained arms 100% candidate accuracy, wrong history 24.4792%.
Retention age/admission, support/opposition, routing, and interaction
gradients nonzero; linear features deliberately zero. No superiority of
literal objective on this constructed task.
Run local tests from repo root:
```sh
PYTHONPATH=tests/nancm_folded:nancm_structural:tests/nancm_retention:tests/nancm_graph python tests/nancm_event_credit/test_event_credit.py
PYTHONPATH=tests/nancm_retention:tests/nancm_graph python tests/nancm_event_credit/test_joint_field.py
```
Corpus continuation is implemented in pilot45.py. It requires successful
v44 retention-metrics.pt and retention-metrics.json, and the exact v36
ancestry recovered by the consolidated workflow. Use v44 command's ancestor
arguments, replace script with tests/nancm_event_credit/pilot45.py, add
--retention results/v44/retention-metrics.pt --retention-metrics
results/v44/retention-metrics.json --out results/v45/metrics.json.
Both matched arms restore the same v44 full Adam state and run 192 further
updates on 12 reference books distinct from the v44 readout fitting subset,
with 12 calibration books and fixed DEV selections. Shared raw mixture
weight is retained from v44, or explicitly forced to .25 if parent weight
was zero (diagnostic only, not parity to parent deployed weight). Calibration
selects lift scale including zero on actual literal BPB for both arms.
The corpus runner has not executed yet. No additional Actions run launched
while the current consolidated recovery and independent diagnostics are active.
Donor, surface factors, and inherited recurrent solver are frozen; gradient
reaches retention and signed edge readout, not all inherited weights. This
is an exact deployed-event credit pathway, not full end-to-end field training.
