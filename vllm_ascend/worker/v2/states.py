import torch

from vllm.v1.worker.gpu.states import RequestState


class AscendRequestState(RequestState):
    """Request state for Ascend NPUs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            max_num_reqs,
            max_model_len,
            max_num_batched_tokens,
            num_speculative_steps,
            vocab_size,
            device,
        )
        self.num_computed_tokens_cpu: torch.Tensor = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        self._pending_request_materialize: dict[int, tuple[int, list[int], int]] = {}

    def add_request(
        self,
        req_id,
        prompt_len,
        all_token_ids,
        num_computed_tokens,
    ) -> None:
        super().add_request(
            req_id,
            prompt_len,
            all_token_ids,
            num_computed_tokens,
        )

        req_idx = int(self.req_id_to_index[req_id])
        self.num_computed_tokens_cpu[req_idx] = num_computed_tokens
        self._pending_request_materialize[req_idx] = (
            int(prompt_len),
            list(all_token_ids),
            int(num_computed_tokens),
        )

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()

        if not self._pending_request_materialize:
            return

        for req_idx, values in self._pending_request_materialize.items():
            prompt_len, all_token_ids, num_computed_tokens = values
            self._materialize_request(
                req_idx,
                prompt_len,
                all_token_ids,
                num_computed_tokens,
            )

        self._pending_request_materialize.clear()

    def _materialize_request(
        self,
        req_idx: int,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None:
        token_len = len(all_token_ids)

        if token_len > 0:
            token_tensor = torch.tensor(
                all_token_ids,
                dtype=self.all_token_ids.gpu.dtype,
                device=self.all_token_ids.gpu.device,
            )
            self.all_token_ids.gpu[req_idx, :token_len] = token_tensor

        self.prompt_len.gpu[req_idx] = prompt_len
        self.prefill_len.gpu[req_idx] = token_len
        self.total_len.gpu[req_idx] = token_len
        self.num_computed_tokens.gpu[req_idx] = num_computed_tokens

        self.prompt_len.np[req_idx] = prompt_len
        self.prefill_len.np[req_idx] = token_len
        self.num_computed_tokens_np[req_idx] = num_computed_tokens
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens_cpu[req_idx] = num_computed_tokens
