# Provenance — NaNCM v178

* Parent recovery ZIP: NaNCM_v177_PREFIX_DAG_MACRO_VERIFIER_RECOVERY_20261123.zip
* Parent SHA-256: 501a360d31be5e63f9da7e480432f640cbbae9e6b5e3b9e0cee39bc76ced5572
* Parent entries: 963; CRC verified prior to extraction. All 963 inherited entries preserved in the v178 release except the root Makefile replaced with an extended copy; original retained as ancestry/v177/Makefile.
* GitHub: shemshallah/nanlm_experimental, master only. Native v178 source same Git blob as GitHub (29e2c8e87b940408986984edc2dbe4b488757e26).
* Native compilation: g++ -O2 -std=c++17 -fPIC -pthread -shared, using original inherited include tree. Linked to no new trained parameters.
* Trained corpus: /results/v175_release3/val.npz reconstructed 48-word contexts, 32 distinct books (chosen by fixed RandomGenerator seed 178); not a fresh unseen test split.
* Source CI run: https://github.com/shemshallah/nanlm_experimental/actions/runs/45250088322
* Full native parity script: experiments/nancm/v178/test_trained178.py
* Cold/warm benchmark: experiments/nancm/v178/bench_cold178.py
* Trained results: results/v178_shared_trained_parity.json, results/v178_cold_warm_trained.json
* No optimizer moments or neural weights updated.
