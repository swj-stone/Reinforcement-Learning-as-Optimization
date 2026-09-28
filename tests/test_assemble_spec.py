"""装配阶段新增常量的契约。

断言刻意是算术恒等式与集合关系，而不是「常量等于我抄的那个数」——
后者在常量被改错时同样通过。
"""

from experiment.data import spec


def test_real_quota_matches_the_plan():
    assert spec.REAL_QUOTA == {"train": 500, "validation": 100, "test": 100}


def test_capacity_cells_put_size_in_the_outer_loop():
    """12 个 cell 的顺序是冻结的产物顺序：n 在外层、alpha 在内层。"""
    assert spec.capacity_cells()[:3] == [(20, 0.25), (20, 0.50), (20, 0.75)]
    assert spec.capacity_cells()[-1] == (200, 0.75)
    assert len(spec.capacity_cells()) == len(spec.SIZES) * len(spec.ALPHAS)


def test_capacity_cells_nesting_differs_from_all_cells():
    """all_cells() 是 family 外层（服务生成的去重顺序），本函数是 n 外层。"""
    assert len(spec.all_cells()) == 48
    # all_cells() 的元素是 (family, n, alpha)：前三个同为 uniform_n20，只有 alpha 在变
    assert [c[2] for c in spec.all_cells()[:3]] == list(spec.ALPHAS)
    assert {c[0] for c in spec.capacity_cells()} == set(spec.SIZES)


def test_stress_quotas_sum_to_the_counts_the_plan_requires():
    """第 8 节：E3 合成 300 + 真实 300；E4 合成 300。"""
    assert spec.E3_SYNTHETIC_PER_FAMILY * len(spec.FAMILIES) * len(spec.ALPHAS) == 300
    assert spec.E3_REAL_PER_ALPHA * len(spec.ALPHAS) == 300
    assert spec.E4_COUNT_PER_ALPHA * len(spec.ALPHAS) == 300


def test_e4_variant_parameters_match_the_plan():
    assert spec.E4_SIZE == 200
    assert spec.E3_SIZE == 500
    assert spec.E4_SKEW_K == 2
    assert spec.E4_SKEW_HEAVY_WEIGHT_RANGE == (500, 2000)
    assert spec.E4_SKEW_HEAVY_DENSITY_RANGE == (100, 200)


def test_seed_tokens_are_outside_the_generation_domains():
    """流隔离的地基。

    种子键的 split 槽位装的是 split 名字、family 槽位装的是族名。一旦某个
    记号与生成阶段的取值重合，真实或应力实例就会与合成实例共用种子域，
    第 8 节「保留独立的 4001/5001 两条流」就名存实亡。
    """
    assert spec.REAL_SEED_TOKEN not in spec.FAMILIES
    assert spec.E4_SEED_TOKEN not in spec.FAMILIES
    assert spec.STRESS_SIZE_STREAM not in spec.SPLITS
    assert spec.STRESS_TAIL_STREAM not in spec.SPLITS
    assert spec.STRESS_SIZE_STREAM != spec.STRESS_TAIL_STREAM


def test_dataset_names_match_the_output_layout():
    assert spec.E3_NAME == "e3_size"
    assert spec.E4_NAME == "e4_tail"


def test_protocol_artifacts_live_at_the_repository_root():
    """datasets/ 被 gitignore，放在那里的审计凭据在新克隆上不存在。"""
    assert spec.DATA_MANIFEST_PATH == spec.REPO_ROOT / "data_manifest.json"
    assert spec.EXPERIMENT_CONFIG_PATH == spec.REPO_ROOT / "experiment_config.yaml"
    assert spec.DATASETS_PROCESSED == spec.REPO_ROOT / "datasets" / "processed"
