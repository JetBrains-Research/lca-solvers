from pipeline.model.adapters.identity_adapter import IdentityAdapter
from pipeline.model.adapters.prefix_unmask_adapter import PrefixUnmaskAdapter
from pipeline.model.adapters.smooth_prefix_unmask_adapter import SmoothPrefixUnmaskAdapter
from pipeline.model.adapters.split_adapter import SplitAdapter

ADAPTERS_REGISTRY = {
    'identity_adapter': IdentityAdapter,
    'prefix_unmask_adapter': PrefixUnmaskAdapter,
    'smooth_prefix_unmask_adapter': SmoothPrefixUnmaskAdapter,
    'split_adapter': SplitAdapter,
}
