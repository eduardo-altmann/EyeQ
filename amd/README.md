# EyeQ em GPUs AMD — Grid'5000 (neowise)

Versão do pipeline para o cluster **neowise** do Grid'5000, em Lyon. Cada nó tem
8 x AMD Radeon Instinct MI50 32 GiB (gfx906), 48 núcleos EPYC e 512 GB de RAM. O
agendador é o OAR.

## Arquivos

| Arquivo | Para quê |
| --- | --- |
| `submit_g5k.sh` | Submete o job noturno (ou o teste de 59 min) no frontend de Lyon |
| `job_g5k.sh` | O job: cópia para `/tmp`, checagem do ROCm, pré-processamento e treino |
| `pyproject.toml`, `uv.lock` | Ambiente Python com PyTorch ROCm, separado do ambiente NVIDIA da raiz |
| `eyeq-rocm.def`, `build_container.sh` | Imagem Apptainer do mesmo ambiente |
| `check_rocm.py` | Checa wheel, GPUs, MIOpen, rocBLAS e RCCL antes do treino |

## O que muda em relação à versão NVIDIA

Tudo abaixo foi encontrado e testado num nó neowise, com o pipeline completo
rodando nas 8 GPUs. As mudanças no código Python só atuam no ROCm, e na NVIDIA
o comportamento continua o mesmo.

- **PyTorch ROCm 6.3.** O ambiente usa `torch 2.9.1+rocm6.3` e Python 3.12. As
  wheels rocm6.4 e 7.x não trazem os kernels gfx906 do rocBLAS, então qualquer
  multiplicação de matrizes falha na MI50.
- **Workers do DataLoader com `forkserver`.** Workers criados por `fork` depois
  da inicialização do RCCL travam o DDP na primeira iteração, com as GPUs
  paradas. O `__main__` de `Main_EyeQuality_tuned_parallel.py` troca o método
  de criação quando o PyTorch é ROCm. Para isso o dataset precisou ser
  serializável: `EyeQ_loader.py` agora recria o transform RGB→LAB em cada worker.
- **`cudnn.benchmark` desligado no ROCm.** Com o benchmark ligado, o MIOpen faz
  tuning de cada convolução por vários minutos na MI50. Sem ele, a primeira
  iteração leva cerca de 40 s e as seguintes 0,5 s.
- **Métricas por classe compatíveis com numpy 2.5.** O relatório final usava
  `float()` em arrays de um elemento, o que o numpy 2.5 não aceita mais. Agora
  usa `.item()`.
- **Job OAR.** `job_g5k.sh` substitui o `.sbatch` do SLURM.

## Antes da primeira execução

1. Commite e envie a pasta `amd/` e as mudanças em `MCF_Net/`. Depois
   atualize o repositório em Lyon:

   ```bash
   ssh lyon.g5k
   cd ~/eyeq/EyeQ && git pull
   ```

2. A imagem Apptainer já foi construída em `~/eyeq/containers/eyeq-rocm6.3.sif`.
   Só refaça se `amd/uv.lock` mudar (o frontend não tem apptainer, então o build
   roda num nó):

   ```bash
   oarsub -t exotic -p "cluster='neowise'" -l host=1,walltime=0:50 \
       "~/eyeq/EyeQ/amd/build_container.sh"
   ```

   Se a imagem não existir, o job monta um venv com uv em `/tmp` a partir do
   mesmo `uv.lock`. Isso baixa cerca de 5 GB e leva uns 3 minutos.

## Execução noturna

No frontend de Lyon:

```bash
~/eyeq/EyeQ/amd/submit_g5k.sh
```

O job pede um nó neowise inteiro com os tipos `night` e `exotic`. No
Grid'5000, o tipo `night` faz o OAR iniciar o job só a partir das 19:00 e
exigir que ele termine até as 09:00. O OAR escolhe a hora de início dentro
dessa janela. Se nenhum nó couber nela, o job não roda naquela noite. O
walltime padrão é 10 horas, e o treino é cortado 20 minutos antes do fim para
dar tempo de copiar os resultados para a home.

Para garantir o nó com antecedência, use uma reserva com hora fixa. O OAR
responde na hora se a reserva é possível:

```bash
~/eyeq/EyeQ/amd/submit_g5k.sh --start "2026-11-06 19:00:00"   # termina às 08:50
```

O padrão é o protocolo `patient_stratified` sem folds, com 8 GPUs, 45 épocas,
batch 4 por GPU e crop 384. Variáveis de ambiente mudam isso:

```bash
EPOCHS=60 GPUS=4 ~/eyeq/EyeQ/amd/submit_g5k.sh --walltime 12:00
~/eyeq/EyeQ/amd/submit_g5k.sh --test     # agora, 59 min, 2 épocas, 1500 imagens por CSV
```

### Tempo esperado

Medido no neowise com 8 GPUs:

| Etapa | Tempo |
| --- | --- |
| Cópia das ~28,8 mil imagens do NFS para `/tmp` | ~11 min |
| Pré-processamento (48 processos, 0,26 s por imagem) | ~3 min |
| Uma época com o dataset inteiro | ~6 a 7 min |
| 45 épocas + avaliação | ~5 a 5,5 h |

Para acompanhar:

```bash
oarstat -u
tail -f ~/eyeq/results/g5k/stratified/<ID>/logs/OAR.*.stdout
oardel <job_id>
# TensorBoard: o log do job mostra o comando do túnel, por exemplo
ssh -L 6006:neowise-3.lyon.grid5000.fr:6006 lyon.g5k   # e abra http://localhost:6006
```

Os resultados ficam em `~/eyeq/results/g5k/stratified/<ID>/`:

- `preprocess/`: métricas do pré-processamento e os CSVs do split usado.
- `training/`: saída de `MCF_Net/result` e `runs/` do TensorBoard.
- `logs/`: saída do OAR, log do pré-processamento, commit e diff do código.

## Pontos de atenção

- **Só as imagens do EyeQ são copiadas.** `original_img` tem o Kaggle DR
  inteiro, com cerca de 88 mil imagens e 90 GB. O EyeQ rotula cerca de 28,8 mil.
  O job copia só as que aparecem nos CSVs. O treino não muda, mas as contagens
  do pré-processamento ficam menores que as da versão SLURM.
- **O batch global muda com o número de GPUs.** `--batch-size` é por GPU e o
  `--lr_scaling` está desligado. Com 8 GPUs e batch 4, o batch global é 32.
  Para comparar com uma execução NVIDIA, use o mesmo produto GPUs x batch.
- **Resultados não são bit a bit iguais** aos da NVIDIA, porque os kernels são
  outros.
- **`rocm-smi` mostra 0% de uso** nas MI50 mesmo com a GPU treinando. Para
  saber se o treino anda, olhe a barra de progresso no `.stderr` do OAR.
- **Cache do MIOpen.** Os kernels compilados voltam para
  `~/eyeq/cache/miopen-rocm6.3` no fim do job. O próximo job começa mais rápido.
- **Imagem Apptainer.** A imagem congela o sistema base (Debian trixie), o
  Python e todos os pacotes. Do host ela usa só o driver `amdgpu`. Ela roda
  sem `--rocm` de propósito, porque essa opção misturaria o ROCm 7.1 do host
  com o ROCm 6.3 da wheel. A base é trixie porque o `fakeroot` do nó exige uma
  glibc mais nova que a do bookworm.

## Se o DDP travar

O sintoma é o treino parado em `Epoch 1`, com os 8 processos a 100% de CPU, até
um timeout do RCCL depois de 10 minutos. No neowise a causa foi o `fork` dos
workers, já corrigido. Para investigar outro caso, `py-spy dump --pid <pid>`
mostra onde cada rank está, e `check_rocm.py --ddp` testa o RCCL isolado.
