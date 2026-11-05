"""
Verifica se o PyTorch enxerga as GPUs AMD antes de gastar horas de fila.

Checa: build ROCm (não CUDA), kernels gfx906 na wheel, número de GPUs, uma
convolução (MIOpen), uma multiplicação de matrizes (rocBLAS) e, com --ddp, um
all_reduce via RCCL entre todas as GPUs.

Uso:
  uv run --project amd python amd/check_rocm.py --gpus 8
  apptainer exec ~/eyeq/containers/eyeq-rocm6.3.sif python amd/check_rocm.py --gpus 8
  uv run --project amd torchrun --standalone --nproc_per_node=8 amd/check_rocm.py --ddp
"""
import argparse
import os
import sys

import torch


def fail(msg):
    print(f'ERRO: {msg}', file=sys.stderr)
    sys.exit(1)


def single(min_gpus):
    print(f'torch {torch.__version__} | HIP {torch.version.hip}')
    if torch.version.hip is None:
        fail('esta wheel do torch não é ROCm (torch.version.hip é None). '
             'Rode com `uv run --project amd`.')
    if not torch.cuda.is_available():
        fail('torch.cuda.is_available() é False. Confira /dev/kfd, /dev/dri e se o '
             'usuário está nos grupos video/render.')

    arch_list = torch.cuda.get_arch_list()
    print(f'Arquiteturas compiladas na wheel: {arch_list}')
    n = torch.cuda.device_count()
    print(f'GPUs visíveis: {n}')
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        arch = getattr(props, 'gcnArchName', '?')
        print(f'  [{i}] {props.name} | {arch} | {props.total_memory / 2**30:.1f} GiB')
        if arch != '?' and arch.split(':')[0] not in arch_list:
            fail(f'GPU {i} é {arch}, mas a wheel não tem kernels para ela.')
    if n < min_gpus:
        fail(f'esperava pelo menos {min_gpus} GPUs, achei {n}.')

    for i in range(n):
        dev = torch.device(f'cuda:{i}')
        x = torch.randn(2, 3, 224, 224, device=dev)
        conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).to(dev)
        y = conv(x).sum()
        y.backward()
        a = torch.randn(512, 512, device=dev)
        (a @ a).sum().item()
    torch.cuda.synchronize()
    print('Convolução (MIOpen) e matmul (rocBLAS) OK em todas as GPUs.')


def ddp():
    import time
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')  # no ROCm, "nccl" usa RCCL
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = f'cuda:{local_rank}'
    # Tamanhos parecidos com os do DDP no MCF-Net: 1 elemento, buffers de
    # BatchNorm (~1,3 MB) e um bucket de gradientes (~100 MB). Mensagens
    # grandes usam outros caminhos do RCCL (P2P/SDMA) que podem travar.
    for numel in (1, 334592, 25_000_000):
        t0 = time.time()
        t = torch.ones(numel, device=dev)
        dist.all_reduce(t)
        b = torch.full((numel,), float(rank), device=dev)
        dist.broadcast(b, src=0)
        torch.cuda.synchronize()
        if int(t[0].item()) != world or float(b[-1].item()) != 0.0:
            fail(f'resultado errado com {numel} elementos')
        if rank == 0:
            print(f'all_reduce + broadcast de {numel * 4 / 2**20:.1f} MB OK em {time.time() - t0:.2f}s')
    if rank == 0:
        print(f'RCCL OK entre {world} processos.')
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='mínimo de GPUs esperado')
    parser.add_argument('--ddp', action='store_true', help='testa all_reduce (rodar via torchrun)')
    args = parser.parse_args()
    if args.ddp:
        ddp()
    else:
        single(args.gpus)
