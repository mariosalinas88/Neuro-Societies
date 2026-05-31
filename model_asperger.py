"""Compatibility wrapper for the simplified model.

Profile id 3 is treated as "Asperger" in the simplified branch, without
rewriting the original profiles.json file. This preserves the original file
while making the working model use the requested label.
"""

from __future__ import annotations

import model_simple as _base


_ORIGINAL_LOAD_PROFILES = _base.load_profiles


def load_profiles(path: str = "profiles.json"):
    profiles = _ORIGINAL_LOAD_PROFILES(path)
    if "3" in profiles:
        profiles["3"]["name"] = "Asperger"
        description = str(profiles["3"].get("description", ""))
        if "Asperger" not in description:
            profiles["3"]["description"] = f"Asperger: {description}"
    return profiles


# Monkey-patch the base module so PsychNeuroSociety uses the alias internally.
_base.load_profiles = load_profiles

PsychAgent = _base.PsychAgent
PsychNeuroSociety = _base.PsychNeuroSociety
TRAIT_KEYS = _base.TRAIT_KEYS
clamp01 = _base.clamp01
gini = _base.gini
sample_profile_traits = _base.sample_profile_traits
