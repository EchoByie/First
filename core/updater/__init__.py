"""The Update section: user-triggered downloads of reference data.

This package is the ONLY network code besides the Ollama backend, and it is
kept apart on purpose: nothing in the run pipeline, the health checks or
the protocols imports it (a test checks this). It runs only when you press
Update in the dashboard or type `aicore update`.
"""
