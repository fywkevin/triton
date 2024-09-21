import pytest
import torch
from torch.profiler import profile

import triton
import triton.language as tl

import os
from contextlib import contextmanager

from dataclasses import dataclass
import numpy as np
import copy

file_name = "attn_wg1_s1.ttgir"
SLOT = 256
WG = 1
exp_attn_kernel = triton.compile(file_name)


@dataclass
class ProfileConfig(object):
    slots: int
    header: int
    wg_num: int
    word_per_slot: int


@dataclass
class Event(object):
    region_id: int
    start: int
    end: int
    wg: int


def get_events(index, wg_id, data):
    actual_slot_num = int(index / 2)
    size = actual_slot_num if len(data) > actual_slot_num else len(data)
    event_list = []
    active_event = {}
    for i in range(0, size, 2):
        metadata = data[i]
        cycle = data[i + 1]
        is_start = metadata >> 31
        region_id = metadata & 0x7FFFFFFF
        if region_id not in active_event:
            active_event[region_id] = Event(region_id, 0, 0, wg_id)
        if is_start == 0:
            active_event[region_id].start = cycle
        else:
            active_event[region_id].end = cycle
            event_list.append(copy.deepcopy(active_event[region_id]))

    return event_list


def shift_start(event_list):
    start_time = []
    for event in event_list:
        if event.start > event.end:
            assert ("Error: cycle overflow, not support for now")
        start_time.append(event.start)
    min_start = min(start_time)
    for event in event_list:
        event.start -= min_start
        event.end -= min_start


def get_scratch_size(config):
    return config.header + config.slots * config.word_per_slot


def get_chrome_event_str(event, block_id, sm_id):
    return f'{{"name": "region_{event.region_id}", "cat": "triton", "ph": "X", "ts": {event.start}, "dur": {event.end - event.start}, "pid": "{block_id}", "tid": "{event.wg}", "args":{{"sm_id": "{sm_id}"}}}}'


def dump_chrome_trace(block_num, config, profile_mem, file_name):
    scratch = get_scratch_size(config)
    trace_str = "{\"traceEvents\": ["
    for i in range(block_num):
        workspace = profile_mem[i * scratch:(i + 1) * scratch]
        block_id = workspace[0].item()
        sm_id = workspace[1].item()
        index = workspace[2].item()
        data = workspace[3:].tolist()
        event_list = []
        wg_data_len = int(len(data) / config.wg_num)
        for j in range(config.wg_num):
            ws_index = j * wg_data_len
            event_list += get_events(index, j, data[ws_index:ws_index + wg_data_len])

        shift_start(event_list)
        for event in event_list:
            chrome_event_str = get_chrome_event_str(event, block_id, sm_id)
            trace_str += chrome_event_str + ",\n"

    trace_str = trace_str[:-2] + "]}"

    with open(file_name, "w") as f:
        f.write(trace_str)


@contextmanager
def set_env_vars(enabled, **kwargs):
    if not enabled:
        yield
        return
    old_values = {}
    for key, value in kwargs.items():
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in kwargs.items():
            del os.environ[key]


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [128]
    for s in ([1] if is_hip() else [3])
    for w in [8]
]

USE_SWP = os.getenv("SWP_FIRST_DOT", "0") == "1"

swp_configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [128]
    for s in ([4] if USE_SWP else [3])
    for w in [8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


#@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        #grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        grid = (128, 64, 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        '''
        # this produces the ttgir file for exp_attn_kernel
        bin = _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            BLOCK_M = 128, #
            BLOCK_N = 128, #
            STAGE=stage,  #
            num_warps=8,  #
            **extra_kern_args)
        '''
        # with profile mem as extra arg and proton.slots
        pconfig = ProfileConfig(slots=SLOT, header=3, wg_num=WG, word_per_slot=2)
        scratch = get_scratch_size(pconfig)
        profile_mem = torch.empty((np.prod(grid) * scratch), device="cuda", dtype=torch.uint32)
        print(q.shape[2])
        bin = exp_attn_kernel[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2),  # q.stride(3), #
            k.stride(0), k.stride(1), k.stride(2),  # k.stride(3), #
            v.stride(0), v.stride(1), v.stride(2),  # v.stride(3), #
            o.stride(0), o.stride(1), o.stride(2),  # o.stride(3), #
            q.shape[0], q.shape[1],  #
            q.shape[2],  #
            #HEAD_DIM=HEAD_DIM_K,  #
            #BLOCK_M = 128, #
            #BLOCK_N = 128, #
            #STAGE=stage,  #
            profile_mem, **extra_kern_args)
        torch.set_printoptions(profile="full")
        print(profile_mem[0:512])
        #dump_chrome_trace(np.prod(grid), pconfig, profile_mem, "chrome_trace.json")
        dump_chrome_trace(20, pconfig, profile_mem, "chrome_trace.json")

        _attn_fwd.kernel = bin
        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply


def run_once():
    BATCH, H, N_CTX, HEAD_DIM = 4, 16, int(2**14), 128
    warmup = 100
    rep = 400
    dtype = torch.bfloat16
    device = "cuda"
    causal = False
    sm_scale = 1.3
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    print("run attention_no_tma")
    res1 = attention(q, k, v, causal, sm_scale)
    print(res1.shape)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    run_once()
