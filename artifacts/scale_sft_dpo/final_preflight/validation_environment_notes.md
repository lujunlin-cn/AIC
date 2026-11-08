# CPU validation scope

Local compileall and the latest complete suite pass (170 Scale / 293 total tests); execution logs are retained in the local validation receipt. The execution host also passes the four added final-evidence tests and the original 166 Scale tests with CUDA hidden.

The additional remote full-suite attempt, before the four added tests, returned 277 passed / 12 failed. Eleven failures require absent ffmpeg/ffprobe executables. The other is the historical LinearScorer padding test requiring bitwise-equal float32 matrix results across different batch shapes. No assertion, inference implementation, dependency stack, or unrelated test was changed to force a pass. Full failure output is retained in validation_cpu_remote.json. These environment/legacy failures do not erase the local full-suite pass or certify the remote full suite.

The required actual Video-ORA smoke evidence predates this bounded run and is reused by SHA256. CPU tests are not described as replacements for real-weight integration.
