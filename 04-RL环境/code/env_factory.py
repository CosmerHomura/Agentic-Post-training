"""
第 4 步 · 环境（environment_factory）
环境 = 一个有状态的世界：能执行动作、记住变化、最后给出一个分数。

TRL 的约定（照做就行）：
  1. 类必须定义在**模块顶层**（第 5 步要 pickle 它到另一个进程）。
  2. __init__(self) 不能带参数（要参数就放进 reset 的 kwargs，来自数据集列）。
  3. reset(**kwargs) 必须存在，返回**观测字符串**（会被当作 prompt），或者 None（表示"我用数据集里的 prompt"）。
  4. 除 reset 之外的**所有公开方法 = 模型可调用的工具**，必须有 docstring，因为 schema 从这里生成。
  5. get_reward() 可选，返回 float；定义了就会自动加到奖励里。
  6. 工具里抛异常不用慌：TRL 会捕获并把异常信息作为工具结果回喂模型（这本身是很好的学习信号）。
"""

import os
import subprocess
import sys
import tempfile


# ============================================================ 例 1：文档里的最小环境
class IncrementEnv:
    """猜数字：模型反复调用 increment，直到命中目标值。

    reset() 自己采样目标值并返回 prompt，所以用这个环境时**不需要外部数据集**。
    """

    def __init__(self):
        self.counter = 0
        self.target = 0

    def reset(self, **kwargs):
        import random
        self.counter = 0
        self.target = random.randint(1, 10)
        return f"数字从 0 开始，每次调用 increment 加一。请把它加到 {self.target}。"

    def increment(self) -> str:
        """把计数器加一。

        Returns:
            加完之后计数器的值。
        """
        self.counter += 1
        return f"counter={self.counter}"

    def get_reward(self) -> float:
        return 1.0 if self.counter == self.target else 0.0


# ============================================================ 例 2：编程环境
class CodingEnv:
    """一个最小的编程环境：读文件 / 写文件 / 跑测试，测试通过才得分。

    ⚠️ 教学版：run_tests 直接在本机 subprocess 里跑 pytest。
       生产环境请把执行放进容器或远程沙箱（OpenEnv + Docker / E2B），
       否则等于让模型在你机器上执行任意代码。
    """

    def __init__(self):
        self.files: dict[str, str] = {}
        self.test_file = "test_solution.py"
        self.workdir = ""

    # ---- 必需：初始化一个 episode ----
    def reset(self, buggy_code: str = "", tests: str = "", **kwargs):
        """接收数据集那一行的其他列，重置环境。

        返回 None = 用数据集里的 prompt 作为第一条用户消息。
        """
        self.files = {"solution.py": buggy_code, self.test_file: tests}
        self.workdir = tempfile.mkdtemp(prefix="coding-env-")
        for name, content in self.files.items():
            with open(os.path.join(self.workdir, name), "w", encoding="utf-8") as f:
                f.write(content)
        return None

    # ---- 以下公开方法都会变成工具 ----
    def read_file(self, path: str) -> str:
        """读取仓库里的一个文件。

        Args:
            path: 文件名，例如 solution.py
        """
        if path not in self.files:
            raise FileNotFoundError(f"没有这个文件：{path}")
        return self.files[path]

    def write_file(self, path: str, content: str) -> str:
        """写入（覆盖）一个文件。

        Args:
            path: 文件名，例如 solution.py
            content: 完整的文件内容
        """
        self.files[path] = content
        with open(os.path.join(self.workdir, path), "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}（{len(content)} 字符）"

    def run_tests(self) -> str:
        """运行 pytest，返回最后几行输出。

        Returns:
            测试输出摘要，包含通过/失败数量。
        """
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", self.test_file],
            cwd=self.workdir, capture_output=True, text=True, timeout=60,
        )
        out = (proc.stdout + proc.stderr).strip()
        return out[-1500:] if out else f"exit code {proc.returncode}"

    # ---- 可选：环境自己给分 ----
    def get_reward(self) -> float:
        """测试全过 = 1.0，否则 0.0（二元奖励通常比塑形奖励更好训）。"""
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", self.test_file],
            cwd=self.workdir, capture_output=True, text=True, timeout=60,
        )
        return 1.0 if proc.returncode == 0 else 0.0


# ============================================================ 多个环境：用 dict 路由
# 数据集里加一列 environment（值为 "increment" / "coding"），TRL 会按名字选 factory。
ENV_FACTORIES = {
    "increment": IncrementEnv,
    "coding": CodingEnv,
}
