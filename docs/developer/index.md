# 开发者指南

欢迎参与 DL-Hub 开发！本节提供项目开发所需的全部参考资料。

---

## 快速导航

<div class="grid cards" markdown>

-   :material-file-tree: **[项目结构](structure.md)**

    ---

    了解仓库目录组织、课程目录规范、输出目录约定

-   :material-source-branch: **[贡献指南](contributing.md)**

    ---

    Fork → Branch → Code → Test → PR 完整流程

-   :material-test-tube: **[测试指南](testing.md)**

    ---

    pytest 使用、冒烟测试、CI 集成

</div>

---

## 核心原则

1. **从零手写** — 使用最少抽象，让学习者看见每一步的细节
2. **统一脚手架** — 所有课程共享 `dlhub/` 提供的训练循环、日志、输出管理
3. **离线可测** — 每个 lesson 必须支持 `--dataset fake` 模式用于冒烟测试
4. **可复现** — 固定随机种子、确定性训练、结果可追踪
