# 云服务器上跑整套流程

[← 总目录](../README.md) · [下一节：SFT →](../01-监督微调-SFT/README.md)

> 这一节的假设：你手上没有 24GB 以上的显卡，只能按小时租云 GPU。
> 目标是用最少的钱把 ①～④ 跑通，需要时再临时上一次双卡摸 ⑤。

---

## 9. 先选机器

| 你要做的事 | 机器 | 大致价格 | 说明 |
|---|---|---|---|
| 跑通 ① SFT、② 蒸馏 | 1×RTX 4090 24G | ¥1.5–2 / 小时 | 0.5–0.6B 全参够用 |
| 跑通 ③ GRPO（小规模） | 1×RTX 4090 24G | 同上 | `num_generations` 降到 4～8 |
| 跑通 ④ 环境（长 episode） | 1×A100 40G | ¥5–6 / 小时 | 8k token 预算不容易爆 |
| ⑤ Harness（模型 ≤1.7B） | 2×RTX 4090 24G | ¥3–4 / 小时 | vLLM 与 trainer 分卡 |
| ⑤ Harness（`Qwen3-4B`） | 1×A800 80G 或 2×A100 | ¥9 / 小时起 | trainer 侧全参要 48–64GB 显存 |

选择的三个要点：

1. **先单卡，再双卡**。前四步单卡全包，别为了"以后要用"多付双卡的钱。
2. **显存优先于算力**。这些实验是显存敏感、算力不敏感；4090 比同价位的 A10 划算。
3. **能用子集就别用全量**。`trl-lib/DeepMath-103K` 全量跑一轮，官方口径是 8×A100 约一天；你裁到 3k 条 prompt，单卡几小时就够看曲线动。

---

## 10. 把代码送上去

三条路，按省事程度排：

### 10.1 git（推荐：改一次推一次，云上 `git pull` 就同步）

本地仓库（第一次）：

```bash
cd "F:\study\AI\Agentic Post-training"     # 或你的实际路径
git init -b main
git add -A
git commit -m "agentic post-training 教程：5 步 + 网页版"
git remote add origin https://github.com/<你的账号>/agentic-post-training.git
git push -u origin main
```

云上：

```bash
git clone https://github.com/<你的账号>/agentic-post-training.git
cd agentic-post-training
```

私有仓库用 HTTPS时要输 **Personal Access Token**（GitHub：Settings → Developer settings → Tokens，勾 `repo`），不是登录密码；或者配 SSH key（`ssh-keygen -t ed25519`，把 `~/.ssh/id_ed25519.pub` 贴进 GitHub → SSH keys）。

### 10.2 scp / rsync（不建仓库，一次拷完）

```bash
rsync -avz --exclude ckpt --exclude .venv --exclude __pycache__ \
  ./ root@<云主机IP>:/root/work/agentic/
```

Windows 没装 rsync 就用 `scp -r`，或者 WinSCP 拖文件夹。

### 10.3 云平台自带的上传（最省事，适合一次性）

AutoDL / 阿里云 PAI / 恒源云的控制台都有 JupyterLab，直接把整个文件夹压成 zip 拖上去，在终端 `unzip` 即可。

### 10.4 仓库里别提交什么

各章目录下的 `ckpt/`（几百 MB 到几十 GB）、`.venv/`、`__pycache__/`、模型缓存、`_site/_preview.html` 这类临时件。工作区根目录已经放了 `.gitignore`，覆盖这些。

> 仓库真正的体积只有约 400KB —— 8 个 README + 16 个示例文件 + 一个 148KB 的单文件网页。`index.html` 一起提交，但不依赖任何托管服务：想要网页版的人自己下载 `index.html`，双击就能看，不需要服务器、不需要联网。
>
> 学习进度（打勾）存在各自浏览器的 localStorage 里，**天然按人分开**：你把 `index.html` 发给十个人，十个人各自记各自的进度，互不影响，也不需要账号。想重来就点页面左下角「重置进度」。

---

## 11. 开机器后 10 分钟

每台新实例照抄这一段。**先看方案 A——它不下载 torch，十分钟内就能开跑。**

### 11.1 方案 A：克隆镜像自带的 base（推荐，零大文件下载）

**先确认 base 里真有 torch**（有些镜像是干净的 miniconda，base 只有 1GB，没装 torch）：

```bash
conda run -n base python -c "import torch;print(torch.__version__, torch.version.cuda)" 2>&1 | tail -2
```

打出版本号才走方案 A；报 `ModuleNotFoundError` 就直接跳到 11.2（但环境仍要建到数据盘，见 11.6）。

确认有 torch 之后，克隆一份 base（硬链接，几秒完成，几乎不占磁盘）：

```bash
conda create -n postrl --clone base -y
conda activate postrl

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "trl>=1.16.0" datasets accelerate openenv-core
```

要补的只有几十 MB 的纯 Python 包，`torch` 一个字节都不用下。

**别写 `trl[vllm]`**：`[vllm]` 会让 pip 重装 vllm，并把它 pin 的 torch 一起换掉，于是又回到 `cuda=False` 那个坑（见 11.4）。

### 11.2 方案 B：空环境自己装 torch（走国内镜像，别用官方源）

如果就是想从零开始：

```bash
conda create -n postrl python=3.11 -y
conda activate postrl

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf

nvidia-smi | head -4        # 右上角 "CUDA Version: 12.8" 就是要匹配的版本
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
```

torch 走官方源在国内往往只有几十 KB/s。下面两个国内镜像都有 cu128 完整目录，`torch-2.11.0+cu128-cp311` 的 wheel 约 820MB：

> **版本上限**：cu128 目前最高只到 **torch 2.11.0**。pip 从 PyPI 装到的 2.13 是 cu130 构建，所以「降到 cu128」必然同时把 torch 版本降到 2.11——这是正常的，不是装错了。

**推荐：把南大索引当 `--index-url` 用**（它是标准 PEP 503 索引，pip 能顺着页面找到 wheel）：

```bash
pip install "torch==2.11.0+cu128" \
  --index-url https://mirror.nju.edu.cn/pytorch/whl/cu128 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

`--extra-index-url` 是用来补 `nvidia-*`、`sympy` 这些依赖的。版本号写死成 `2.11.0+cu128`，pip 就不会去拿清华源里更高版本的 cu130 包。

**如果直链下 403**：阿里云的 `pytorch-wheels` 会对部分 IP 直接返回 `403 Forbidden`（pip 和 curl 都一样），这不是你写错了。两个办法：

```bash
# 办法一：走南大直链（同一份文件）
wget -c "https://mirror.nju.edu.cn/pytorch/whl/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl" \
  -O "/root/torch-2.11.0+cu128.whl"
pip install "/root/torch-2.11.0+cu128.whl" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 办法二：带浏览器 UA 从阿里云下（403 多半是 UA/防盗链判定）
curl -L -C - -A "Mozilla/5.0" -o "/root/torch-2.11.0+cu128.whl" \
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
pip install "/root/torch-2.11.0+cu128.whl" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

先本地存成 `.whl` 再 `pip install 文件` 还有个好处：断了能续传，不会每次重下 820MB。

还嫌慢就上多线程下载器：`aria2c -x16 -s16 <上面的 url> -o /root/torch.whl`，下完 `pip install /root/torch.whl`。

镜像可用性（2026-10 实测）：

| 镜像 | `pytorch-wheels/cu128` | 说明 |
|---|---|---|
| 阿里云 `mirrors.aliyun.com/pytorch-wheels/` | ✅ 200 / ⚠️ 403 | 目录可浏览；**部分 IP 直链下 wheel 会 403**，见上面两个办法 |
| 南大 `mirror.nju.edu.cn/pytorch/whl/` | ✅ 200 | 完整 PEP 503 索引，可直接当 `--index-url` 用（推荐） |
| 上交 `mirror.sjtu.edu.cn` | ⚠️ 301 | 能进，但要跟随跳转，且没有 `torch/` 子索引 |
| 清华 / 南大 / 北外 / CERNET / 中科大 | ❌ 404 | 只镜像 anaconda 与 PyPI，没有 pytorch-wheels |

按驱动选目录：

| `nvidia-smi` 右上角 | 用哪个目录 |
|---|---|
| CUDA Version: 12.8 | `https://mirror.nju.edu.cn/pytorch/whl/cu128/` |
| 12.4 | `.../cu124/` |
| 12.1 | `.../cu121/` |
| 13.0 以上 | `.../cu130/` |

装错的典型报错（**这条不是你代码的问题**）：

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080) ... torch 2.13.0+cu130  cuda=False
```

意思是 torch 是 cu130 编的、驱动只到 12.8。换成对应目录的 wheel 重装即可，不用动驱动：

```bash
# 0) 先看清楚现在装了什么：vllm 与 torch 是强绑定，换 torch 前必须先确认
pip list | grep -Ei '^(torch|vllm|trl|transformers) '
pip show vllm 2>/dev/null | grep -i requires     # 它会写死 torch==某个版本

# 1) 换 wheel（就地修，不用删环境）
pip uninstall -y torch torchvision torchaudio
pip install "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"   # 要 True
pip check          # 看依赖有没有被这次替换弄坏
```

如果上一步发现 vllm 写死了别的 torch 版本，两条路选一条：把 vllm 一起换到匹配你 wheel 的版本，或者**先把 vllm 卸掉**——第 1、2 步根本用不到它，等到第 3～5 步再装。

**要 vllm 的话最后再装**（它只用于第 3～5 步加速）：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple vllm`，装完**重跑 11.3 自检**——vllm 会把 torch 换成它 pin 的版本，如果又变成高版本 CUDA 编的，就按上面的办法把同版本号的 cu128 wheel 装回去。

第 4 步的 `environment_factory` 要求 `transformers>=5.2.0`，④ 报版本不够时再单独升一行 `pip install -U "transformers>=5.2.0"`。

### 11.3 装完自检（两条路线都跑这三行）

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers, trl;print(transformers.__version__, trl.__version__)"
python -c "from transformers import AutoTokenizer;t=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B');print('template ok', len(t.chat_template))"
```

第一行必须是 `True`；`torch.version.cuda` 要 ≤ `nvidia-smi` 那个版本号。

第 4 步的 `environment_factory` 要求 `transformers>=5.2.0`，④ 报版本不够时再单独升一行 `pip install -U "transformers>=5.2.0"`。

新终端会丢两样东西（环境和镜像），一次性写进 `~/.bashrc` 就不必每次重敲：

```bash
echo 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate postrl' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/hf' >> ~/.bashrc
```

漏掉第一行的症状是 `pip install` 装进了 base（`which python` 指到 `/usr/bin` 或 conda base），漏掉后两行的症状是模型下载 `ConnectionError` / `Timeout`。

### 11.4 为什么必须建环境

1. **`trl[vllm]` 会顺手换掉 torch**。vLLM 对 torch 版本卡得很死，pip 装它时会覆盖镜像里那份预编译的 torch+CUDA，症状是自检里 `torch.cuda.is_available()` 变成 `False`，或 NCCL报 `unhandled cuda error`。真遇到了就**降 vllm 版本去匹配镜像 torch**，别反过来升 torch。
2. **依赖会打架**：教程要较新的 `transformers`，你别的项目可能锁在旧版本。
3. **实例一换就没了**，有 conda 环境 + `requirements.txt`（`pip freeze > requirements.txt`）才谈得上复现。

方案 B 就是这条路：代价是几 GB 的 torch 下载（走阿里云镜像大约几分钟）；好处是完全隔离，同机跑多个互斥项目也不打架。不想付这个代价就用方案 A 的 `conda create --clone base`，隔离性一样有，下载量接近零。

本地 Windows / macOS 同样用 conda：`conda create -n postrl python=3.11 -y && conda activate postrl`。

### 11.5 AutoDL 学术加速：开与关

AutoDL 的「学术加速」是一个只会对 **github.com / githubusercontent.com / githubassets.com / huggingface.co** 生效的代理，用来解决 `git clone` 和直连 HF 慢的问题。

```bash
# 开（只对当前这个终端有效，新开终端要重新执行）
source /etc/network_turbo

# 关
unset http_proxy && unset https_proxy
```

两点要注意：

1. **它是环境变量代理**：`source` 之后 `env | grep -i proxy` 能看到 `http_proxy` / `https_proxy`，新终端失效需重新 `source`。
2. **只在需要 `git clone` 时开，用完就关**：官方说过它会影响正常网络，留着它跑 `pip install`（清华源）或 `scp` 反而变慢甚至失败；拉模型走 ② 的 `HF_ENDPOINT` 镜像就够了，两个别同时开。

如果执行 `source /etc/network_turbo` 报 `No such file or directory`，说明你所在区域不支持该代理，那就走镜像站 + `ghproxy.link`（地址会变，进站点看当前可用的）。


### 11.6 报 `[Errno 28] No space left on device` 怎么办

装 torch 时最常见，不是网络问题，是**系统盘（`/root` 那 30G）被填满**了。
一个 cu128 的 torch 要：wheel 本体 820MB + 解压后 2.6GB + 一堆 `nvidia-*-cu12` 依赖约 4GB + pip 缓存 1～2GB ≈ **7～8GB**。

`pip` 的缓存往往是最大的一块：装过几次 torch，`/root/.cache/pip` 轻松到 10GB 以上（每次 820MB 的 wheel 都会留一份）。所以**第一步永远是清缓存**。

先看清楚谁在吃盘：

```bash
df -h / /root/autodl-tmp              # 系统盘 vs 数据盘
du -sh /root/* 2>/dev/null | sort -h | tail -6
du -sh /root/miniconda3/envs/*        # 是不是环境建在系统盘了
du -sh "$(pip cache dir)" 2>/dev/null # pip 缓存
```

三条治法，按顺序试：

```bash
# 1) 清缓存 + 卸掉装了一半的 torch
pip cache purge
conda clean -a -y
rm -rf /root/.cache/pip /root/tmp/*.whl
pip uninstall -y torch torchvision torchaudio

# 2) 新环境直接建到数据盘（路径用 -p 指定，不走 miniconda3/envs）
conda create -p /root/autodl-tmp/envs/postrl python=3.11 -y
conda activate /root/autodl-tmp/envs/postrl

# 3) 缓存和临时目录也搬到数据盘，再装（--no-cache-dir 不再留副本）
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
pip install --no-cache-dir "torch==2.11.0+cu128" \
  --index-url https://mirror.nju.edu.cn/pytorch/whl/cu128 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

盘上的预算大致是：torch 全家桶解压后约 6.5GB（其中 4GB 是 nvidia 的库），模型缓存 `HF_HOME=/root/autodl-tmp/hf` 另算（0.6B 约 1.2GB，1.7B 约 3.4GB）。

最省的做法还是 **11.1 的克隆 base**：镜像自带的 torch 本来就按这张卡的驱动编好，`--clone` 用硬链接，几秒完成且几乎不占额外空间，一个字节的 wheel 都不用下。


---

## 12. 每一步怎么跑

### 12.1 先验证管线，再真训练

`data/` 里的 jsonl 只有 2～3 条，是给你**解剖数据格式**用的，不是训练数据。
所以每条命令分两段：先小跑验证管线，再换官方数据集真训。

```bash
# 全部在仓库根目录执行（不用 cd 进 code/）
python 01-监督微调-SFT/code/train_sft.py --inspect          # 只看数据链路，不需 GPU
python 01-监督微调-SFT/code/train_sft.py --dataset trl-lib/Capybara --limit 2000 --eval-size 100

python 02-蒸馏/code/train_distill.py --inspect
python 02-蒸馏/code/train_distill.py --dataset trl-lib/ultrafeedback-prompt --limit 3000 --eval-size 100

python 03-强化学习-GRPO/code/train_grpo.py --inspect
python 03-强化学习-GRPO/code/train_grpo.py --dataset trl-lib/DeepMath-103K --limit 3000 --eval-size 100

python 04-RL环境/code/train_grpo_env.py
```

脚本按自己的位置算出所在章节，所以**从哪个目录启动都能找到 `data/`**，产物也只会落在本章的 `runs/` 与 `ckpt/`（例如 `01-监督微调-SFT/runs/sft/`），不会往仓库根堆。
加 `--dataset` 后数据从 Hugging Face 下载，记得 `export HF_ENDPOINT=https://hf-mirror.com`。

每章「怎么判断真的学会了」看哪个指标，README 里都有一张表（SFT 看 held-out -ln p、蒸馏看 KL、GRPO看 pass@1）。

### 12.2 用 tmux 防断线

网页终端一断，前台进程就死。所有超过 5 分钟的任务都这样开：

```bash
tmux new -s train          # 新建会话
python 03-强化学习-GRPO/code/train_grpo.py     # 在里面跑
# Ctrl+B 然后 D 脱离；下次 tmux attach -t train 回到现场
```

临时任务用 `nohup python 03-强化学习-GRPO/code/train_grpo.py > log.txt 2>&1 &` 也行。

### 12.3 双卡（⑤ Harness）

开两个 tmux 窗口：

```bash
# 窗口 1：vLLM 生成服务，占 0 号卡
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
vllm serve Qwen/Qwen3-0.6B \
  --max-model-len 4096 \
  --logprobs-mode processed_logprobs \
  --weight-transfer-config '{"backend":"nccl"}'
```

```bash
# 窗口 2：训练，占 1 号卡（参数仍是仓库根目录下的脚本路径）
CUDA_VISIBLE_DEVICES=1 accelerate launch 05-Harness与规模化/code/train_async_grpo.py
```

脚本里的 `vllm_server_url` 默认就是 `http://localhost:8000`（见 `train_async_grpo.py` 顶部注释），换成别的端口要同步改。

### 12.4 训练时看 TensorBoard（AutoDL 的 AutoPanel）

AutoDL 实例详情页右侧有个 **AutoPanel**，里面的 TensorBoard 只读一个固定目录：`/root/tf-logs`。
我们的脚本默认不写 TensorBoard（`report_to=[]`），要看就加 `--tensorboard`：

```bash
# 1）确认有 tensorboard（镜像一般自带；没有就装，几 MB）
python -c "import tensorboard, sys; print('ok', tensorboard.__version__)"

# 2）开训时加 --tensorboard，日志落在 <章节>/runs/<x>/tb/
python 01-监督微调-SFT/code/train_sft.py \
  --dataset trl-lib/Capybara --limit 2000 --eval-size 100 --tensorboard

# 3）把它挂到 AutoPanel 认的目录（软链接，不占空间、不动代码）
mkdir -p /root/tf-logs
ln -sfn /root/Agentic-Post-training/01-监督微调-SFT/runs/sft/tb /root/tf-logs/sft
```

然后：**AutoDL 控制台 → 实例 → AutoPanel → TensorBoard → 打开**。它会自动拉起一个 TensorBoard
进程（默认 6006 端口），页面里就能看到 `loss`、`learning_rate`、`grad_norm`、`eval/loss` 等曲线，
横轴是 step，可以缩放、可以对比多次运行。

跑第 2、3 章时各挂一个子目录，TensorBoard 会当成不同的 run 并排比较：

```bash
ln -sfn /root/Agentic-Post-training/02-蒸馏/runs/distill/tb /root/tf-logs/distill
ln -sfn /root/Agentic-Post-training/03-强化学习-GRPO/runs/grpo/tb   /root/tf-logs/grpo
```

看不到东西时按顺序查：

| 现象 | 原因 | 处理 |
|---|---|---|
| 面板一直「等待日志」 | 目录挂错了 | `ls -l /root/tf-logs/`，应能看到 `sft -> .../runs/sft/tb` |
| 目录空的 | 没加 `--tensorboard`，或还没跑满 1 步 | 加了重跑；`logging_steps=1`，第 1 步就有数据 |
| 页面能开但没曲线 | AutoPanel 缓存了旧的空目录 | 面板里点重启 TensorBoard，或换个子目录名 |

两种看法不冲突：`--tensorboard` 给浏览器看交互曲线，`--plot-every 20` 顺手写一张 `loss.svg` 快照
（断线、离线、发群里都方便）。只想留一个就 `--plot-every 0` 或去掉 `--tensorboard`。

想在本地看（不经过 AutoPanel）：先 `ssh -L 6006:localhost:6006 root@<ip>`，然后在云端跑
`tensorboard --logdir 01-监督微调-SFT/runs/sft/tb --port 6006`，浏览器开 `http://localhost:6006`。

---

## 13. 云上四个坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 下载模型 `ConnectionError` / `Timeout` | 没设 `HF_ENDPOINT`，或设了但没重开终端 | `export HF_ENDPOINT=https://hf-mirror.com`，写进 `~/.bashrc` |
| 第 5 步 NCCL 初始化卡住或报 P2P 错 | 消费级 4090 走 PCIe，没有 NVLink/P2P | 能跑但慢：把 `weight_sync_steps` 调大（少同步几次）；实在不行换 A100/A800 |
| `openenv` / Harbor 连不上沙箱 | e2b、Daytona 要注册并绑卡 | 学习阶段先用 `sandbox=None` + 本地 Docker，只保留"完整 agent 循环"这一点 |
| 磁盘写满，训练中途报 `No space left` | 系统盘 30G，模型 + checkpoint 一下就满 | `HF_HOME` 指到数据盘；`save_strategy` 别用 `steps` 高频存 |

排错时先看这三条：

```bash
df -h                                  # 磁盘
nvidia-smi                             # 显存有没有被别人占、是不是掉卡
tail -f log.txt                        # 有没有 HF_ENDPOINT 相关的报错
```

---

## 14. 花多少钱

按上面的价格，一个人完整走一遍的账单大概是这样：

| 阶段 | 机器 | 时长 | 费用 |
|---|---|---|---|
| ①～④ 跑通示例（本仓库的小数据） | 1×4090 | 1–2 小时 | ¥3–5 |
| ①～④ 换成自己的真数据（几千条） | 1×4090 | 6–10 小时 | ¥15–25 |
| ⑤ 双卡跑 0.6B/1.7B，摸通异步流程 | 2×4090 | 6–8 小时 | ¥25–35 |
| ⑤ 双卡 A100 + 4B 模型 | 2×A100 | 1 天 | ¥250–400 |

省钱的四个习惯：

1. **先小后大**：先用示例数据确认脚本能跑通，再换真数据；别一上来就全量数据集。
2. **随时存 checkpoint**：`save_strategy="epoch"` + 手动 `trainer.save_model()`，机器被回收也不白跑。
3. **用完就关**：按量计费关机不再收 GPU 费（只收少量存储费），别让它空转一夜。
4. **模型选小的**：`Qwen3-0.6B` 和 `Qwen2.5-0.5B` 足以看清 loss、奖励、`mean_token_accuracy` 的变化趋势；4B 只在你想验证"规模化后行为是否还一致"时再上。

---

## 15. 上线前检查清单

- [ ] `nvidia-smi` 能看到卡，显存 ≥ 24GB（双卡时两张都在）
- [ ] `HF_ENDPOINT` 已设且在 `~/.bashrc` 里
- [ ] `HF_HOME` 指向数据盘，`df -h` 剩余 ≥ 50GB
- [ ] `trl`、`transformers` 版本符合（第 4 步要 `transformers>=5.2.0`）
- [ ] 在 `code/` 目录里能读到 `../data/*.jsonl`
- [ ] 长任务都在 `tmux` 里
- [ ] 知道 checkpoint 会存在哪、怎么拉回本地（`scp -r root@<ip>:/path/ckpt .`）
- [ ] 知道这次跑完大概花多少钱、什么时候该关机

---

[← 总目录](../README.md) · [下一节：SFT →](../01-监督微调-SFT/README.md)
