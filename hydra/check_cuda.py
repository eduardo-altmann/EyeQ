"""
Verifica se o PyTorch enxerga a GPU NVIDIA antes de gastar tempo de reserva.

Checa: build CUDA, kernels para a arquitetura da GPU (sm_90 na GH200), número
de GPUs, uma convolução (cuDNN) e uma multiplicação de matrizes (cuBLAS) e,
com --ddp, um all_reduce via NCCL.

Uso:
  apptainer exec --nv eyeq-cuda12.8-aarch64.sif python hydra/check_cuda.py --gpus 1
  apptainer exec --nv eyeq-cuda12.8-aarch64.sif torchrun --standalone --nproc_per_node=1 hydra/check_cuda.py --ddp
"""
import argparse
import os
import sys

import torch


def fail(msg):
    print(f'ERRO: {msg}', file=sys.stderr)
    sys.exit(1)


def single(min_gpus):
    print(f'torch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}')
    if torch.version.cuda is None:
        fail('esta wheel do torch não é CUDA (torch.version.cuda é None).')
    if not torch.cuda.is_available():
        fail('torch.cuda.is_available() é False. A imagem foi executada com --nv? '
             'O driver do nó é novo o bastante (>= 570 para CUDA 12.8)?')
    arch_list = torch.cuda.get_arch_list()
    print(f'Arquiteturas compiladas na wheel: {arch_list}')
    n = torch.cuda.device_count()
    print(f'GPUs visíveis: {n}')
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        sm = f'sm_{props.major}{props.minor}'
        print(f'  [{i}] {props.name} | {sm} | {props.total_memory / 2**30:.1f} GiB')
        if sm not in arch_list:
            fail(f'GPU {i} é {sm}, mas a wheel não tem kernels para ela.')
    if n < min_gpus:
        fail(f'esperava pelo menos {min_gpus} GPUs, achei {n}.')
    for i in range(n):
        dev = torch.device(f'cuda:{i}')
        x = torch.randn(2, 3, 224, 224, device=dev)
        conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).to(dev)
        conv(x).sum().backward()
        a = torch.randn(512, 512, device=dev)
        (a @ a).sum().item()
    torch.cuda.synchronize()
    print('Convolução (cuDNN) e matmul (cuBLAS) OK em todas as GPUs.')


def ddp():
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    world = dist.get_world_size()
    t = torch.ones(1, device=f'cuda:{local_rank}')
    dist.all_reduce(t)
    torch.cuda.synchronize()
    if int(t[0].item()) != world:
        fail('all_reduce com resultado errado')
    if dist.get_rank() == 0:
        print(f'NCCL OK entre {world} processos.')
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='mínimo de GPUs esperado')
    parser.add_argument('--ddp', action='store_true', help='testa all_reduce (rodar via torchrun)')
    args = parser.parse_args()
    ddp() if args.ddp else single(args.gpus)
