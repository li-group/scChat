# @generated from torch/_C/return_types.pyi

from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from torch import contiguous_format, Generator, inf, memory_format, strided, Tensor, SymInt
from torch.types import (
    _bool,
    _device,
    _dtype,
    _float,
    _int,
    _layout,
    _qscheme,
    _size,
    Number,
)

class _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(NamedTuple):
    output: Tensor
    mask: Tensor

class _fused_moving_avg_obs_fq_helper(NamedTuple):
    output: Tensor
    mask: Tensor

class _linalg_det(NamedTuple):
    result: Tensor
    LU: Tensor
    pivots: Tensor

class _linalg_eigh(NamedTuple):
    eigenvalues: Tensor
    eigenvectors: Tensor

class _linalg_slogdet(NamedTuple):
    sign: Tensor
    logabsdet: Tensor
    LU: Tensor
    pivots: Tensor

class _linalg_solve_ex(NamedTuple):
    result: Tensor
    LU: Tensor
    pivots: Tensor
    info: Tensor

class _linalg_svd(NamedTuple):
    U: Tensor
    S: Tensor
    Vh: Tensor

class _lu_with_info(NamedTuple):
    LU: Tensor
    pivots: Tensor
    info: Tensor

class _scaled_dot_product_efficient_attention(NamedTuple):
    output: Tensor
    log_sumexp: Tensor
    philox_seed: Tensor
    philox_offset: Tensor

class _scaled_dot_product_flash_attention(NamedTuple):
    output: Tensor
    logsumexp: Tensor
    cum_seq_q: Tensor
    cum_seq_k: Tensor
    max_q: Union[_int, SymInt]
    max_k: Union[_int, SymInt]
    philox_seed: Tensor
    philox_offset: Tensor
    debug_attn_mask: Tensor

class _unpack_dual(NamedTuple):
    primal: Tensor
    tangent: Tensor

class aminmax(NamedTuple):
    min: Tensor
    max: Tensor

class cummax(NamedTuple):
    values: Tensor
    indices: Tensor

class cummin(NamedTuple):
    values: Tensor
    indices: Tensor

class frexp(NamedTuple):
    mantissa: Tensor
    exponent: Tensor

class geqrf(NamedTuple):
    a: Tensor
    tau: Tensor

class histogram(NamedTuple):
    hist: Tensor
    bin_edges: Tensor

class histogramdd(NamedTuple):
    hist: Tensor
    bin_edges: List[Tensor]

class kthvalue(NamedTuple):
    values: Tensor
    indices: Tensor

class lu_unpack(NamedTuple):
    P: Tensor
    L: Tensor
    U: Tensor

class max(NamedTuple):
    values: Tensor
    indices: Tensor

class median(NamedTuple):
    values: Tensor
    indices: Tensor

class min(NamedTuple):
    values: Tensor
    indices: Tensor

class mode(NamedTuple):
    values: Tensor
    indices: Tensor

class nanmedian(NamedTuple):
    values: Tensor
    indices: Tensor

class qr(NamedTuple):
    Q: Tensor
    R: Tensor

class slogdet(NamedTuple):
    sign: Tensor
    logabsdet: Tensor

class sort(NamedTuple):
    values: Tensor
    indices: Tensor

class svd(NamedTuple):
    U: Tensor
    S: Tensor
    V: Tensor

class topk(NamedTuple):
    values: Tensor
    indices: Tensor

class triangular_solve(NamedTuple):
    solution: Tensor
    cloned_coefficient: Tensor
