"""Human-readable name generation for job IDs."""

import random


_ADJECTIVES = [
    'bold', 'bright', 'calm', 'crisp', 'deft', 'eager', 'fair', 'fleet',
    'gentle', 'grand', 'keen', 'light', 'lucid', 'mild', 'neat', 'noble',
    'plain', 'prime', 'proud', 'pure', 'quick', 'quiet', 'rare', 'sharp',
    'sleek', 'smart', 'smooth', 'solid', 'stark', 'steady', 'still',
    'stout', 'strong', 'subtle', 'sure', 'swift', 'terse', 'true',
    'vivid', 'warm', 'wise', 'young', 'amber', 'azure', 'coral', 'dusky',
    'fern', 'frost', 'ivory', 'jade', 'lunar', 'moss', 'opal', 'pearl',
    'russet', 'sage', 'scarlet', 'silver', 'slate', 'tawny', 'violet',
]

_VERBS = [
    'arcing', 'binding', 'coiling', 'docking', 'ebbing', 'folding',
    'gliding', 'homing', 'joining', 'keying', 'lacing', 'mapping',
    'nesting', 'orbiting', 'packing', 'racing', 'rising', 'roaming',
    'scanning', 'seeking', 'soaring', 'sorting', 'spinning', 'staging',
    'striding', 'surfing', 'tracing', 'turning', 'vaulting', 'wading',
    'winding', 'arming', 'blazing', 'carving', 'crossing', 'darting',
    'diving', 'drifting', 'fading', 'flaring', 'forging', 'grazing',
    'hunting', 'landing', 'leaping', 'mending', 'pacing', 'probing',
    'ranging', 'roving', 'sailing', 'scaling', 'sifting', 'sparking',
    'steering', 'surging', 'tapping', 'threading', 'tracking', 'waking',
]

_NOUNS = [
    'anvil', 'arch', 'basin', 'beacon', 'blade', 'bolt', 'cairn',
    'cedar', 'cliff', 'comet', 'condor', 'crest', 'delta', 'drift',
    'dune', 'falcon', 'fern', 'finch', 'flint', 'forge', 'frost',
    'grove', 'gull', 'hawk', 'heath', 'heron', 'isle', 'jade',
    'larch', 'ledge', 'linden', 'marten', 'mesa', 'mica', 'moss',
    'newt', 'oak', 'orca', 'osprey', 'otter', 'peak', 'pine',
    'plover', 'quartz', 'rapids', 'raven', 'reef', 'ridge', 'robin',
    'sage', 'shoal', 'slate', 'sparrow', 'spruce', 'stone', 'swift',
    'talon', 'tern', 'thorn', 'tide', 'vale', 'wren',
]


def generate_readable_name() -> str:
    """Generate a human-readable adjective-verbing-noun tag.

    Returns names like 'swift-folding-falcon' or 'calm-spinning-quartz'.
    Combination space: ~60 x 60 x 60 = ~216,000 unique names.
    """
    return (
        f"{random.choice(_ADJECTIVES)}-"
        f"{random.choice(_VERBS)}-"
        f"{random.choice(_NOUNS)}"
    )
