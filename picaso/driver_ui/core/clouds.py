"""
Clouds panel. Cloud N is stored as `clouds.cloudN_type` (the chosen type) and
`clouds.cloudN.<type>` (parameters for every available type).
"""
import copy
import re

from picaso.driver_ui.core.config_schema import Section, Selector, build_fields

_CLOUD_KEY = re.compile(r"cloud(\d+)$")


def cloud_ids(clouds):
    """cloud1, cloud2, ... in numeric order."""
    numbered = [(int(m.group(1)), key) for key in clouds if (m := _CLOUD_KEY.match(key))]
    return [key for _, key in sorted(numbered)]


def fields(clouds):
    children = []
    for cloud_id in cloud_ids(clouds):
        cloud = clouds[cloud_id]
        cloud_type = clouds.get(f"{cloud_id}_type")
        types = list(cloud)
        if cloud_type not in types:
            cloud_type = types[0]
        branch = build_fields(cloud[cloud_type], ("clouds", cloud_id, cloud_type))
        selector = Selector(("clouds", f"{cloud_id}_type"), cloud_type, types, branch)
        children.append(Section(("clouds", cloud_id), [selector]))
    return Section(("clouds",), children)


def resize(clouds, count):
    """Adds clouds (copied from cloud1, first type selected) or removes the highest-numbered ones."""
    ids = cloud_ids(clouds)
    for cloud_id in ids[count:]:
        del clouds[cloud_id]
        clouds.pop(f"{cloud_id}_type", None)
    for i in range(len(ids) + 1, count + 1):
        clouds[f"cloud{i}"] = copy.deepcopy(clouds["cloud1"])
        clouds[f"cloud{i}_type"] = next(iter(clouds["cloud1"]))
