import pytest
import torch
from torch.profiler import profile

import triton
import triton.language as tl

import os
from contextlib import contextmanager
import numpy as np
from proton_chrome_trace import ProfileConfig, get_scratch_size, dump_chrome_trace

INTRA_PROFILE = True
run_kernel = ["attention_tma", "attention_no_tma"]

if "attention_no_tma" in run_kernel:
    file_name_attn = "attn.ttgir"
    SLOT_attn = 256
    WG_attn = 1
    exp_attn_kernel = triton.compile(file_name_attn)
    chrome_trace_attn = "chrome_trace_attn.json"

if "attention_tma" in run_kernel:
    file_name_attn_tma = "attn_tma.ttgir"
    SLOT_attn_tma = 256
    WG_attn_tma = 2
    exp_attn_tma_kernel = triton.compile(file_name_attn_tma)
    chrome_trace_attn_tma = "chrome_trace_attn_tma.json"


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
def _attn_fwd_inner_tma(acc, l_i, m_i, q,  #
                        K_desc_ptr, V_desc_ptr, Q, qvk_offset, stride_kn, stride_vn, stride_vk,  #
                        start_m, qk_scale,  #
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                        N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    # Required TMA fences are added in _attn_fwd_tma prior to calling this function.
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl._experimental_descriptor_load(  # load in row major
            K_desc_ptr,
            [start_n.to(tl.int32) + (qvk_offset // stride_kn).to(tl.int32), 0],
            [BLOCK_N, HEAD_DIM],
            Q.dtype.element_ty,
        )
        k = tl.trans(k)
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
        if fp8_v:
            v = tl._experimental_descriptor_load(  # load in row major
                V_desc_ptr,
                [(qvk_offset // stride_vn).to(tl.int32), start_n.to(tl.int32)],
                [HEAD_DIM, BLOCK_N],
                Q.dtype.element_ty,
            )
            v = tl.trans(v)
        else:
            v = tl._experimental_descriptor_load(  # load in row major
                V_desc_ptr,
                [(qvk_offset // stride_vk + start_n).to(tl.int32), 0],
                [BLOCK_N, HEAD_DIM],
                Q.dtype.element_ty,
            )
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_tma(  #Q, V, desc_k, desc_v, sm_scale, M, Out,  #
        Q, V, Out, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
        stride_qz, stride_qh, stride_qm, stride_qk,  #
        stride_kz, stride_kh, stride_kn, stride_kk,  #
        stride_vz, stride_vh, stride_vk, stride_vn,  #
        stride_oz, stride_oh, stride_om, stride_on,  #
        Z, H, N_CTX,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        HEAD_DIM: tl.constexpr,  #
        STAGE: tl.constexpr  #
):
    # TODO(embg) remove TMA fence after __grid_constant__ lands
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_q], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_k], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_v], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [desc_o], dtype=tl.int32, is_pure=False, pack=1)

    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

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
    q = tl._experimental_descriptor_load(  # load in row major
        desc_q,
        [(qvk_offset // stride_qm + start_m * BLOCK_M).to(tl.int32), 0],
        [BLOCK_M, HEAD_DIM],
        Q.dtype.element_ty,
    )
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q, desc_k, desc_v, Q, qvk_offset, stride_kn, stride_vn,
                                            stride_vk,  #
                                            start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner_tma(acc, l_i, m_i, q, desc_k, desc_v, Q, qvk_offset, stride_kn, stride_vn,
                                            stride_vk,  #
                                            start_m, qk_scale,  #
                                            BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                            2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                            )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl._experimental_descriptor_store(desc_o, acc.to(Out.type.element_ty),
                                      [(qvk_offset // stride_om + start_m * BLOCK_M).to(tl.int32), 0])


def alloc_descs(q, k, v):
    BATCH, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
    TMA_SIZE = 128
    META = {"BLOCK_M": 128, "BLOCK_N": 128}
    o = torch.empty_like(q)
    desc_k = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
    desc_v = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
    desc_q = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
    desc_o = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
    q_buf = torch.empty_like(desc_q, device="cpu", pin_memory=True)
    k_buf = torch.empty_like(desc_k, device="cpu", pin_memory=True)
    v_buf = torch.empty_like(desc_v, device="cpu", pin_memory=True)
    o_buf = torch.empty_like(desc_o, device="cpu", pin_memory=True)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(k.data_ptr(), BATCH * H * N_CTX, HEAD_DIM_Q,
                                                              META['BLOCK_N'], HEAD_DIM_Q, k.element_size(),
                                                              k_buf.data_ptr())
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(v.data_ptr(), BATCH * H * N_CTX, HEAD_DIM_Q,
                                                              META['BLOCK_N'], HEAD_DIM_Q, v.element_size(),
                                                              v_buf.data_ptr())
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(q.data_ptr(), BATCH * H * N_CTX, HEAD_DIM_Q,
                                                              META['BLOCK_M'], HEAD_DIM_Q, q.element_size(),
                                                              q_buf.data_ptr())
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(o.data_ptr(), BATCH * H * N_CTX, HEAD_DIM_Q,
                                                              META['BLOCK_M'], HEAD_DIM_Q, o.element_size(),
                                                              o_buf.data_ptr())
    desc_q.copy_(q_buf, non_blocking=True)
    desc_k.copy_(k_buf, non_blocking=True)
    desc_v.copy_(v_buf, non_blocking=True)
    desc_o.copy_(o_buf, non_blocking=True)
    return o, desc_q.cuda(), desc_k.cuda(), desc_v.cuda(), desc_o.cuda()


def attention_tma(q, k, v, o, desc_q, desc_k, desc_v, desc_o, causal, sm_scale):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-2] if v.dtype == torch.float8_e5m2 else v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    stage = 3 if causal else 1
    extra_kern_args = {}

    BATCH, H, N_CTX = q.shape[0], q.shape[1], q.shape[2]

    # no autotune with fixed BLOCK_N
    def grid_tma(META):
        return (triton.cdiv(q.shape[2], 128), q.shape[0] * q.shape[1], 1)

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    grid = (128, 64, 1)

    if not INTRA_PROFILE:
        bin = _attn_fwd_tma[grid](
            q, v, o, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            HEAD_DIM=HEAD_DIM_K,  #
            BLOCK_M=128,  #
            BLOCK_N=128,  #
            STAGE=stage,  #
            num_warps=8,  #
            **extra_kern_args)
    else:
        pconfig = ProfileConfig(slots=SLOT_attn_tma, header=3, wg_num=WG_attn_tma, word_per_slot=2)
        scratch = get_scratch_size(pconfig)
        profile_mem = torch.empty((np.prod(grid) * scratch), device="cuda", dtype=torch.uint32)
        bin = exp_attn_tma_kernel[grid](
            q, v, o, desc_q, desc_k, desc_v, sm_scale, M, desc_o,  #
            q.stride(0), q.stride(1), q.stride(2),  #q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2),  #k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2),  #v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2),  #o.stride(3),  #
            q.shape[0], q.shape[1],  #
            q.shape[2],  #
            #HEAD_DIM=HEAD_DIM_K,  #
            #BLOCK_M = 128, #
            #BLOCK_N = 128, #
            #STAGE=4,  #
            profile_mem, **extra_kern_args)
        torch.set_printoptions(profile="full")
        print(profile_mem[0:512])
        torch.set_printoptions(profile="default")
        # dump_chrome_trace(np.prod(grid), pconfig, profile_mem, chrome_trace_attn_tma)
        dump_chrome_trace(20, pconfig, profile_mem, chrome_trace_attn_tma)

    _attn_fwd_tma.kernel = bin
    return o


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

        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        grid = (128, 64, 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        if not INTRA_PROFILE:
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
                BLOCK_M=128,  #
                BLOCK_N=128,  #
                STAGE=stage,  #
                **extra_kern_args)
        else:
            # with profile mem as extra arg and proton.slots
            pconfig = ProfileConfig(slots=SLOT_attn, header=3, wg_num=WG_attn, word_per_slot=2)
            scratch = get_scratch_size(pconfig)
            profile_mem = torch.empty((np.prod(grid) * scratch), device="cuda", dtype=torch.uint32)
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
            torch.set_printoptions(profile="default")
            # dump_chrome_trace(np.prod(grid), pconfig, profile_mem, chrome_trace_attn)
            dump_chrome_trace(20, pconfig, profile_mem, chrome_trace_attn)

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
    # causal = True
    causal = False
    sm_scale = 1.3
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    if "attention_no_tma" in run_kernel:
        print("run attention_no_tma")
        res1 = attention(q, k, v, causal, sm_scale)
        print(res1)

    if "attention_tma" in run_kernel:
        print("run attention_tma")
        o, desc_q, desc_k, desc_v, desc_o = alloc_descs(q, k, v)
        res2 = attention_tma(q, k, v, o, desc_q, desc_k, desc_v, desc_o, causal, sm_scale)
        print(res2)

    if "attention_no_tma" in run_kernel and "attention_tma" in run_kernel:
        print("mean diff between attention_no_tma and attention_tma")
        print(torch.abs(res1 - res2).mean())


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    run_once()
