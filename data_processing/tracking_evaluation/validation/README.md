# Official-reference verification

`trackeval_reference_verification.json` records a successful comparison of the independent HOTA/IDF1 implementation with the official TrackEval sources at commit `12c8791b303e0a0b50f753af204249e622d0281a`. It pins the evaluator module hash, downloaded source hashes, dependency versions, test seed, comparisons and numeric tolerance. This is a metric correctness check on generated observations, separate from the real light example.

The preserved `verify_reference.py` harness expects an `official/` directory beside it containing the pinned files listed in the verification JSON, plus `official/provenance.json` equal to that JSON's `reference` object. It imports the existing repository at `/home/zhengpengen/gdc_atrium`. For another checkout, supply its package on `PYTHONPATH` and adapt that location in a new copy of the harness.

For a fresh reproduction, use a new temporary directory and an isolated environment with NumPy and SciPy. Copy the harness there, download the four pinned files to their recorded relative paths, verify their SHA-256 hashes, and write the provenance object. Preserve the official MIT license beside the downloaded sources. Run the copied harness with the isolated Python interpreter; it produces a new `verification.json` beside itself. The application and normal evaluator need neither these downloads nor NumPy/SciPy.

The official metric source bodies are unchanged. Harness-only compatibility shims replace timing/config wrappers and legacy `np.float`/`np.int` aliases. They do not alter the HOTA or identity calculations. Both the standard-library and SciPy assignment paths are tested against the same official reference. Empty IDF1 follows an explicitly documented null-versus-zero convention.
