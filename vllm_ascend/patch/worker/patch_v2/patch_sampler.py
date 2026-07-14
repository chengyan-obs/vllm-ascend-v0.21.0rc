from vllm.v1.worker.gpu.sample import gumbel, sampler, states

from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample

sampler.gumbel_sample = gumbel_sample
gumbel.apply_temperature = apply_temperature
states.apply_temperature = apply_temperature
