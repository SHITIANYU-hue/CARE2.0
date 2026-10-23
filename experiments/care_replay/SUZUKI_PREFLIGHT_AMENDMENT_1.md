# Suzuki 冻结前结构核验：修订 1 待决策

用户已批准原方案中的 BH→Suzuki、10 对生成及 R1/R2、最多 20 次模型调用和 Suzuki 单独 $1 上限。授权记录见 `outputs/suzuki_second_dataset_preflight_v1/authorization_and_status.json`。此次未要求重新批准相同资源。

**状态：正式冻结前停止，API 调用 0、费用 $0、评估轨迹 0。** 没有改动生产代码、共享描述符 CSV 或历史实验。原方案／准备清单保留原字节。

原方案要求“已有可解析值沿用”，同时要求冻结前核验真实分子身份。核验发现以下三项可解析 SMILES 与供应商分子身份不一致，需要扩展修正范围。

| 配体（各 480 行） | 旧代码结构得到的 MW | 供应商结构经本机 RDKit 得到的 MW | 对可执行类别的影响 |
|---|---:|---:|---|
| AmPhos | 305.361 | 265.381 | MW 与 logP 桶不变，但分子结构错误应修正 |
| CataCXium A | 222.312 | 358.550 | mw_mid→mw_high；logp_high→logp_very_high |
| XPhos | 434.648 | 476.729 | mw_high→mw_very_high；logP 桶不变 |

这些是三个另外的配体，不是原方案中缺失结构的 Xantphos/dppf/dtbpf。新发现错误结构对应 1,440 行（25%）；其中 960 行（16.67%）会改变现用语义分箱。结合原缺失结构，至少一半反应行的配体结构表征需要缺失处理或更正；不表示反应产率或行本身错误。

来源与复算：

- [Merck AmPhos／APhos，CAS 932710-63-9](https://www.merckmillipore.com/NC/en/product/sial/pmc2010)：供应商结构含两个 tert-butyl，旧代码含两个 phenyl；分子式 C16H28NP 对 C20H20NP。
- [Sigma-Aldrich CataCXium A，CAS 321921-71-5](https://www.sigmaaldrich.com/AW/en/product/aldrich/671479)：供应商结构为二金刚烷基正丁基膦，旧代码为二叔丁基苯基膦。
- [Sigma-Aldrich XPhos，CAS 564483-18-7](https://www.sigmaaldrich.com/US/en/product/aldrich/638064)：供应商结构有三个异丙基，旧代码只有两个。
- [Sigma-Aldrich SPhos](https://www.sigmaaldrich.com/IN/en/product/aldrich/638072)：另作一致性核对，供应商结构与旧代码规范化后相同。

来源查阅日期 2026-10-20。MW 的小数末位与供应商展示有轻微差异，是用同一本机 RDKit 对结构重算；logP 是同一 Crippen 算法计算值，不是声称供应商报告的实验 logP。原／参考 SMILES、CAS、URL、分子式、计算值、分箱和 RDKit 版本见 `structure_identity_audit.json`。当前仅完成上述四个命名配体的身份对照，不将它说成全体化合物核验已经完成。

## 推荐的具体修订

1. 在独立 Suzuki 表征层中，按上述可追溯结构更正 AmPhos、CataCXium A、XPhos，重算衍生 MW／logP；共享 CSV 和旧冻结来源不变。不将新数值回写到旧 BH 结果。
2. 原三种缺失结构仍标 `unknown`，无配体仍为 `none`，角色规范化和重复字段处理按已批准方案继续。
3. **沿用原分箱界限**：MW 120/250/450；logP 0/2/5。不合并类别、不调整阈值来维持原来可交换组。
4. 修订后 MW 为 5 类，logP 为 4 类，故原本 MW↔logP 的 R2 字段互换不再合法。保留“同类型且同基数”的 R2 定义，只允许 boolean 两字段之间的互换；其余为单例组。R2 仍可随机合法值和符号，但字段扰动明显更受限，报告必须明确。
5. 数据、规则数、每库五选一、覆盖容差、gate、三项主对比、预算和调用数都按原批准方案；生成前完成其余结构／身份核验、离线测试，再冻结。

修订后的 public-only 目录预览：

| 字段 | 类别与行数 |
|---|---|
| ligand_mw_bin | mw_mid 480；mw_high 2880；mw_very_high 480；none 480；unknown 1440 |
| ligand_logp_bin | logp_high 1920；logp_very_high 1920；none 480；unknown 1440 |

这是未实施的字段投影，没有读取目标产率来选分箱，也没有调用模型。完整拟定目录和 R2 类型组见 `amendment_1_proposed_catalog.json`。

## 待用户决定

**是否批准这三项可核验结构更正，并接受 R2 按原同类型／同基数约束自然缩小可交换范围？** 推荐批准。另一可行选择是暂把这三个身份错误的配体 MW／logP 也标为 unknown，保留原四类结构，但这会让 50% 的行缺少这两项数值信息；不建议继续使用已知错误描述符，也不建议为了扩大 R2 而改变分箱。

按照用户“需要改方案的地方停下来问我”的要求，当前只保存审计和修订预览，未应用表征更正、冻结或开始 API。批准后继续执行原已授权流程；若其余核验发现新的科学定义变化，另行报告。
