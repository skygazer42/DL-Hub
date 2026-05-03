# Topic Coverage

DL-Hub now keeps the broad community topic pool in a machine-checkable coverage
registry instead of relying on README claims alone.

The registry lives in `dlhub/topic_coverage.py` and maps every requested topic to
one or more concrete code artifacts. It covers three shapes of topic:

- model or task directions, such as detection, segmentation, 3D point cloud,
  video understanding, deraining, OCR, and geo-localization
- research workflow streams, such as paper digests, resource sharing,
  open-source projects, surveys, datasets, and tutorials
- cross-cutting method/framework topics, such as NAS, AutoML, pruning,
  distillation, SLAM, metaverse scene assets, PyTorch, TensorFlow, MXNet,
  TensorRT, OpenCV, NumPy, and Python

The runtime helpers are:

```python
from dlhub.topic_coverage import coverage_report, describe_topic, validate_topic_coverage

report = coverage_report()
validate_topic_coverage()
print(describe_topic("目标检测").primary_artifact.module)
```

Additional lightweight surfaces:

- `dlhub/research_streams.py` for paper/resource/survey/dataset/tutorial topics
- `dlhub/framework_adapters.py` for optional framework probes without importing
  heavyweight packages
- `dlhub/method_kits.py` for runnable NAS/AutoML, pruning, distillation, SLAM,
  capsule-routing, and scene-asset utilities

Regression coverage is in `tests/test_topic_coverage.py`. The test checks that
every topic in the requested pool has an existing path and an importable module,
then exercises lookup, stream, framework, and method-kit behavior.
