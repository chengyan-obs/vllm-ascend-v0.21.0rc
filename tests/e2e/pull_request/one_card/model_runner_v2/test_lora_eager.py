import os
from unittest.mock import patch

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_port


REQUIRED_PATH_VARS = (
    "V2_EAGER_LORA_MODEL_PATH",
    "V2_EAGER_LORA_ALICE_PATH",
    "V2_EAGER_LORA_BOB_PATH",
)

pytestmark = pytest.mark.skipif(
    any(not os.environ.get(name) for name in REQUIRED_PATH_VARS),
    reason="requires local Qwen3-0.6B, Alice, and Bob paths",
)


def _completion(client, model: str) -> str:
    response = client.completions.create(
        model=model,
        prompt="Hello, my name is",
        max_tokens=8,
        temperature=0,
    )
    text = response.choices[0].text
    assert text.strip(), f"{model}: empty completion"
    return text


def test_qwen3_eager_static_lora_switch() -> None:
    base = os.environ["V2_EAGER_LORA_MODEL_PATH"]
    alice = os.environ["V2_EAGER_LORA_ALICE_PATH"]
    bob = os.environ["V2_EAGER_LORA_BOB_PATH"]
    server_args = [
        "--served-model-name",
        "qwen3-0.6b",
        "--trust-remote-code",
        "--generation-config",
        "vllm",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.8",
        "--enforce-eager",
        "--enable-lora",
        "--lora-modules",
        f"Alice={alice}",
        f"Bob={bob}",
        "--max-loras",
        "2",
        "--max-cpu-loras",
        "4",
        "--max-lora-rank",
        "8",
    ]
    env_dict = {
        "ASCEND_LAUNCH_BLOCKING": "1",
        "VLLM_USE_MODELSCOPE": "True",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
    }

    server_port = get_open_port()
    server_args.extend(["--port", str(server_port)])

    with patch.dict(
        os.environ,
        {
            "NO_PROXY": "127.0.0.1,localhost,0.0.0.0",
            "no_proxy": "127.0.0.1,localhost,0.0.0.0",
        },
        clear=False,
    ):
        with RemoteOpenAIServer(
            base,
            server_args,
            server_host="127.0.0.1",
            server_port=server_port,
            auto_port=False,
            env_dict=env_dict,
            max_wait_seconds=180,
        ) as server:
            client = server.get_client()
            model_ids = {model.id for model in client.models.list().data}
            assert {"qwen3-0.6b", "Alice", "Bob"} <= model_ids

            alice_first = _completion(client, "Alice")
            bob_text = _completion(client, "Bob")
            alice_second = _completion(client, "Alice")

    assert alice_first != bob_text, (
        "Alice and Bob requests returned the same adapter output"
    )
    assert alice_first == alice_second, (
        "Alice output changed after the Bob request"
    )
