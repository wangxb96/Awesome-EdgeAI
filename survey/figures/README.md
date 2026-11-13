# Editable figures

The diagrams use a shared blue, teal and amber palette. Pictograms denote conceptual roles, not tested products or vendor logos. Arrows describe dependencies and control flow; their lengths do not encode performance.

| Figure | Reading purpose | Source | Vector export |
|---|---|---|---|
| Resource map | Entry point from task acceptance to deployment and evidence | [Python](resource_map.py) | [PDF](resource_map.pdf), [SVG](resource_map.svg) |
| Cognitive edge overview | Task loop, resource policy and device–edge–cloud execution | [Python](cognitive_edge_overview.py) | [PDF](cognitive_edge_overview.pdf), [SVG](cognitive_edge_overview.svg) |
| Memory budget | Separate fixed weights from growing KV state | [Python](memory_budget.py), [analytical CSV](memory_budget_analytical.csv) | [PDF](memory_budget.pdf), [SVG](memory_budget.svg) |
| Quantization fidelity | Nine task-score values reported in QSpec Table 1, not new measurements | [Python](quantization_fidelity.py), [source CSV](quantization_fidelity_source.csv) | [PDF](quantization_fidelity.pdf), [SVG](quantization_fidelity.svg) |
| Placement decision | Local, delegated and deferred execution conditions | [Python](deployment_reasoning.py) | [PDF](placement_decision.pdf), [SVG](placement_decision.svg) |
| Collaboration patterns | Payload, KV ownership and state boundaries | [Python](deployment_reasoning.py) | [PDF](collaboration_patterns.pdf), [SVG](collaboration_patterns.svg) |
| Cross-layer couplings | Shared memory, compute and workflow amplification | [Python](deployment_reasoning.py) | [PDF](cross_layer_couplings.pdf), [SVG](cross_layer_couplings.svg) |
| Workflow evaluation | Request timing and failure-aware task accounting | [Python](deployment_reasoning.py) | [PDF](workflow_evaluation.pdf), [SVG](workflow_evaluation.svg) |
| Assistant workflow | Permission, execution and recovery boundaries | [Python](assistant_workflow.py) | [PDF](assistant_workflow.pdf), [SVG](assistant_workflow.svg) |

The memory panel is formula-generated with stated model assumptions; it is not a measured device-fit claim. The quantization panel replots reported QSpec exact-match scores for one stated model/quantizer setting; the source table supplies no uncertainty estimates. Other figures are conceptual schematics. The assistant is a prospective synthesis example, not an executed application benchmark. Dashed task-quality feedback in the overview can be offline or delayed validation.

## Rebuild

The tested environment uses Python 3.14 with [pinned matplotlib and NumPy packages](../requirements-figures.txt). The scripts use standard Matplotlib primitives and no private assets. Install the packages with `python3 -m pip install -r survey/requirements-figures.txt`. The scripts resolve output paths relative to their own directory; run from the repository root:

```sh
python3 survey/figures/resource_map.py
python3 survey/figures/cognitive_edge_overview.py
python3 survey/figures/memory_budget.py
python3 survey/figures/quantization_fidelity.py
python3 survey/figures/deployment_reasoning.py
python3 survey/figures/assistant_workflow.py
```

`resource_map.py` and `assistant_workflow.py` reuse the included vector primitives in `deployment_reasoning.py`. Exports include editable PDF/SVG, PNG previews and 600-dpi TIFF. Collision-audit JSON files accompany the rendered figures; a geometry audit is additional to visual and scientific review.

[Back to resource map](../../README.md)
