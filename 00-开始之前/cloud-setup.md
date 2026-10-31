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
| ⑤ Harness（`Qwen3-4B`） | 1×A800 80G 或 2×A100 | ¥9 / 小时 起 | trainer 侧全参要 48–64GB 显存 |

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

私有仓库用 HTTPS 时要输 **Personal Access Token**（GitHub：Settings → Developer settings → Tokens，勾 `repo`），不是登录密码；或者配 SSH key（`ssh-keygen -t ed25519`，把 `~/.ssh/id_ed25519.pub` 贴进 GitHub → SSH keys）。

### 10.2 scp / rsync（不建仓库，一次拷完）

```bash
rsync -avz --exclude ckpt --exclude .venv --exclude __pycache__ \
  ./ root@<云主机IP>:/root/work/agentic/
```

Windows 没装 rsync 就用 `scp -r`，或者 WinSCP 拖文件夹。

### 10.3 云平台自带的上传（最省事，适合一次性）

AutoDL / 阿里云 PAI / 恒源云的控制台都有 JupyterLab，直接把整个文件夹压成 zip 拖上去，在终端 `unzip` 即可。

### 10.4 仓库里别提交什么

`ckpt/`（几百 MB 到几十 GB）、`.venv/`、`__pycache__/`、模型缓存、`_site/_preview.html` 这类临时件。工作区根目录已经放了 `.gitignore`，覆盖这些。

> 仓库真正的体积只有约 400KB —— 8 个 README + 16 个示例文件 + 一个 148KB 的单文件网页。`index.html` 一起提交，但不依赖任何托管服务：想要网页版的人自己下载 `index.html`，双击就能看，不需要服务器、不需要联网。
>
> 学习进度（打勾）存在各自浏览器的 localStorage 里，**天然按人分开**：你把 `index.html` 发给十个人，十个人各自记各自的进度，互不影响，也不需要账号。想重来就点页面左下角「重置进度」。

---

## 11. 开机器后 10 分钟

每台新实例照抄这一段，四步走完就能开跑。

```bash
# (1) conda 环境（为什么必须建，见 11.1）
conda create -n postrl python=3.11 -y
conda activate postrl

# (2) 镜像与缓存（国内直连 huggingface.co 基本必失败）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf          # 系统盘通常只有 30G，模型别放那儿

# (3) 依赖（清华源，装进 postrl 环境）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "trl>=1.16.0" transformers datasets accelerate trl[vllm] openenv-core

# (4) 自检，三行都要过
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers, trl;print(transformers.__version__, trl.__version__)"
python -c "from transformers import AutoTokenizer;t=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B');print('template ok', len(t.chat_template))"
```

第 4 步的 `environment_factory` 要求 `transformers>=5.2.0`，④ 报版本不够时再单独升一行 `pip install -U "transformers>=5.2.0"`。

新终端会丢两样东西（环境和镜像），一次性写进 `~/.bashrc` 就不必每次重敲：

```bash
echo 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate postrl' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/hf' >> ~/.bashrc
```

漏掉第一行的症状是 `pip install` 装进了 base（`which python` 指到 `/usr/bin` 或 conda base），漏掉后两行的症状是模型下载 `ConnectionError` / `Timeout`。

### 11.1 为什么必须建环境

1. **`trl[vllm]` 会顺手换掉 torch**。vLLM 对 torch 版本卡得很死，pip 装它时会覆盖镜像里那份预编译的 torch+CUDA，症状是上面 ④ 里 `torch.cuda.is_available()` 变成 `False`，或 NCCL 报 `unhandled cuda error`。真遇到了就**降 vllm 版本去匹配镜像 torch**，别反过来升 torch。
2. **依赖会打架**：教程要较新的 `transformers`，你别的项目可能锁在旧版本。
3. **实例一换就没了**，有 conda 环境 + `requirements.txt`（`pip freeze > requirements.txt`）才谈得上复现。

conda 建的是空环境，torch 要按镜像的 CUDA 版本重装一份（`pip install torch --index-url https://download.pytorch.org/whl/cu121` 之类），代价是几 GB 下载；好处是完全隔离，同机跑多个互斥项目也不打架。

本地 Windows / macOS 同样用 conda：`conda create -n postrl python=3.11 -y && conda activate postrl`。

### 11.2 AutoDL 学术加速：开与关

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


---

## 12. 每一步怎么跑

### 12.1 先验证管线，再真训练

`data/` 里的 jsonl 只有 2～3 条，是给你**解剖数据格式**用的，不是训练数据。
所以每条命令分两段：先小跑验证管线，再换官方数据集真训。

```bash
cd 01-监督微调-SFT/code
python train_sft.py --inspect                      # 只看数据链路，不需 GPU
python train_sft.py --dataset trl-lib/Capybara --limit 2000 --eval-size 100

cd 02-蒸馏/code
python train_distill.py --inspect
python train_distill.py --dataset trl-lib/ultrafeedback-prompt --limit 3000 --eval-size 100

cd 03-强化学习-GRPO/code
python train_grpo.py --inspect
python train_grpo.py --dataset trl-lib/DeepMath-103K --limit 3000 --eval-size 100

cd 04-RL环境/code
python train_grpo_env.py
```

脚本里的数据路径是 `../data/xxx.jsonl`，所以**必须先进到 `code/` 目录**再执行；在仓库根目录跑会报文件不存在。
加 `--dataset` 后数据从 Hugging Face 下载，记得 `export HF_ENDPOINT=https://hf-mirror.com`。

每章「怎么判断真的学会了」看哪个指标，README 里都有一张表（SFT 看 held-out -ln p、蒸馏看 KL、GRPO 看 pass@1）。

### 12.2 用 tmux 防断线

网页终端一断，前台进程就死。所有超过 5 分钟的任务都这样开：

```bash
tmux new -s train          # 新建会话
python train_grpo.py       # 在里面跑
# Ctrl+B 然后 D 脱离；下次 tmux attach -t train 回到现场
```

临时任务用 `nohup python train_grpo.py > log.txt 2>&1 &` 也行。

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
# 窗口 2：训练，占 1 号卡
cd 05-Harness与规模化/code
CUDA_VISIBLE_DEVICES=1 accelerate launch train_async_grpo.py
```

脚本里的 `vllm_server_url` 默认就是 `http://localhost:8000`（见 `train_async_grpo.py` 顶部注释），换成别的端口要同步改。

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
