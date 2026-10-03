"""试验种子的派生（设计 §4）。"""

import subprocess
import sys

from experiment.baselines.seeds import trial_seed
from experiment.data.spec import REPO_ROOT

#: 全部期望值由探针实测得到（probe2.py，2026-09-30）。
INSTANCE_ID = "syn-test-uniform-n20-a25-i0000"
BASE = 6001


def test_the_seed_is_the_measured_literal():
    """钉住算法本身：BLAKE2b（8 字节）、键 ``base|id|trial``、大端。"""
    assert trial_seed(INSTANCE_ID, 0, BASE) == 13096480787329443436
    assert trial_seed(INSTANCE_ID, 1, BASE) == 11617834981237195108
    assert trial_seed(INSTANCE_ID, 9, BASE) == 14240787302913443677


def test_seed_is_stable_across_processes():
    """**跨进程**稳定：全新解释器里给出同一个种子。

    **它断言的字面量与 ``test_the_seed_is_the_measured_literal`` 是同一个**，所以
    它不是那个字面量的独立证据：那一条已经足以挡下 ``hash()`` 实现——``hash(str)``
    每个进程用不同的随机盐，进程内的取值同样不可能等于写死的字面量（那一条测试
    会红，靠的是这个，不是本条的跨进程比较）。

    它真正多提供的是：派生在**独立进程**里成立，即不依赖本进程内积累的任何状态
    ——模块级计数器、别的测试留下的 ``random.seed()``、导入顺序。

    子进程的 ``cwd`` 设成仓库根目录，``python -c`` 才会把 ``experiment`` 放进
    ``sys.path``。
    """
    code = (
        "import sys\n"
        "from experiment.baselines.seeds import trial_seed\n"
        "sys.stdout.write(str(trial_seed(%r, 0, %d)))\n" % (INSTANCE_ID, BASE)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=True,
        cwd=str(REPO_ROOT),
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout == "13096480787329443436"


def test_every_component_changes_the_seed():
    """三个分量各自都进哈希。

    少进一个分量（比如忘了 ``trial``）的实现，会让十次试验抽到同一串硬币、
    十行数字完全相同——而上面两条测试仍然通过。
    """
    base = trial_seed(INSTANCE_ID, 0, BASE)
    assert trial_seed(INSTANCE_ID, 1, BASE) != base            # trial 进了
    assert trial_seed(INSTANCE_ID, 0, BASE + 1) != base        # base 进了
    assert trial_seed(INSTANCE_ID + "x", 0, BASE) != base      # instance_id 进了


def test_the_seed_uses_all_eight_digest_bytes():
    """种子用满 64 位，而不是只用低 32 位。

    若 ``digest_size`` 被改成 4，所有种子都小于 ``2**32``。3,300 实例 × 10 次
    试验 = 33,000 次抽取，在 2**32 空间里的生日碰撞概率约 **12%**，在 2**64**
    里约 3e-11；碰撞意味着两次"独立"试验抽到同一串硬币，取平均时它们是重复
    计数。

    用 1,000 次抽取的**最大值**而不是单个字面量：单个字面量只能证明这一个种子，
    最大值的量级才说明整个分布用满了空间。
    """
    seeds = [
        trial_seed(f"syn-test-uniform-n20-a25-i{i:04d}", trial, BASE)
        for i in range(100)
        for trial in range(10)
    ]
    assert len(seeds) == 1000
    assert max(seeds) >= 2**32
