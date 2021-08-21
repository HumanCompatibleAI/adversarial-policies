from collections import OrderedDict
import functools
import json
import logging
import os
import re
import sys
from typing import Any, Iterable, Sequence, Tuple

import boto3

from aprl.visualize import util

logger = logging.getLogger("aprl.visualize.generate_website_json")

ENV_NAME_LOOKUP = {
    "KickAndDefend-v0": "Kick and Defend",
    "SumoHumans-v0": "Sumo Humans",
    "YouShallNotPassHumans-v0": "You Shall Not Pass",
    "SumoAnts-v0": "Sumo Ants",
}
BUCKET_NAME = "adversarial-policies-public"
PREFIX = "videos"

EXCLUDE_ABBREV = [r"ZooM[SD].*"]


class NestedDict(OrderedDict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def get_s3_files() -> Iterable[str]:
    s3 = boto3.resource("s3")
    adv_policies_bucket = s3.Bucket(BUCKET_NAME)
    objs = adv_policies_bucket.objects.filter(Prefix=PREFIX).all()
    return [os.path.basename(o.key) for o in objs]


def is_excluded(abbrev: str) -> bool:
    for exclude in EXCLUDE_ABBREV:
        if re.match(exclude, abbrev):
            return True
    return False


def get_videos(video_files: Iterable[str]) -> NestedDict:
    video_files = [path for path in video_files if path.endswith(".mp4")]
    stem_pattern = re.compile(r"(.*)_[0-9]+p.mp4")
    agent_pattern = re.compile(r"(\w*-v\d)_victim_(.*)_opponent_(.*)")

    nested = NestedDict()
    for path in video_files:
        stem_match = stem_pattern.match(path)
        if stem_match is None:
            logger.info(f"Skipping path '{path}: malformed filename, cannot extract stem.")
            continue

        stem = stem_match.groups()[0]
        assert isinstance(stem, str)

        agent_match = agent_pattern.match(stem)
        if agent_match is None:
            logger.info(f"Skipping path '{path}: malformed filename, cannot extract agent.")
            continue

        env_name, victim_abbrev, opponent_abbrev = agent_match.groups()
        if is_excluded(victim_abbrev) or is_excluded(opponent_abbrev):
            logger.info(f"Skipping path '{path}': explicitly excluded.")
            continue

        env_name = ENV_NAME_LOOKUP.get(env_name)
        victim = f"{util.friendly_agent_label(victim_abbrev)} ({victim_abbrev})"
        opponent = f"{util.friendly_agent_label(opponent_abbrev)} ({opponent_abbrev})"
        nested[env_name][opponent][victim] = stem

    return nested


def sort_fn(item: Tuple[str, Any], groups: Sequence[str]) -> str:
    """Prepends index of key in groups: can sort in order of groups with alphabetical sort.

    :param item: key-value pair.
    :param groups: sequence of regexps."""
    k, v = item
    match = re.match(r".* \((.*)\)", k)
    assert match
    abbrev = match.groups()[0]
    for i, grp in enumerate(groups):
        if re.match(grp, abbrev):
            break
    return f"{i}{abbrev}"


def sort_nested(nested: NestedDict) -> NestedDict:
    new_nested = NestedDict()
    victim_sort = functools.partial(sort_fn, groups=util.GROUPS["rows"])
    opponent_sort = functools.partial(sort_fn, groups=util.GROUPS["cols"])

    for env, d1 in nested.items():
        new_d1 = {}
        for opponent, d2 in d1.items():
            new_d1[opponent] = OrderedDict(sorted(d2.items(), key=victim_sort))
        new_nested[env] = OrderedDict(sorted(new_d1.items(), key=opponent_sort))
    return new_nested


def main():
    logging.basicConfig(level=logging.INFO)
    paths = get_s3_files()
    nested = get_videos(paths)
    nested = sort_nested(nested)

    out_path = "file_list.json"
    if len(sys.argv) > 1:
        out_path = sys.argv[1]

    print(nested)
    with open(out_path, "w") as fp:
        json.dump(nested, fp, indent=4)
    logger.info(f"Saved files to '{out_path}'.")


if __name__ == "__main__":
    main()
