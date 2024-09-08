from pipeline.model.adapters.identity_adapter import IdentityAdapter
from pipeline.model.adapters.prefix_unmask_adapter import PrefixUnmaskAdapter

ADAPTERS_REGISTRY = {
    'identity_adapter': IdentityAdapter,
    'prefix_unmask_adapter': PrefixUnmaskAdapter,
}
