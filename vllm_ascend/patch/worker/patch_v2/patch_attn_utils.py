import vllm

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache as _ascend_allocate_kv_cache,
    _reshape_kv_cache as _ascend_reshape_kv_cache,
)


def _allocate_kv_cache_compat(kv_cache_config, device):
    return _ascend_allocate_kv_cache(kv_cache_config, {}, device)


vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache_compat
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _ascend_reshape_kv_cache
