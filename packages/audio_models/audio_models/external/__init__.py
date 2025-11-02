"""Utility namespace for managing external audio model repositories.

This package hosts helper modules that clone and interface with the upstream
research repositories referenced in ``requirements.csv``.  At runtime the
wrappers under ``audio_models`` call into these helpers to make sure the
original code is available before delegating inference to it.
"""
