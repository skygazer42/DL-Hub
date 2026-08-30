import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _constructor_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _keyword_bool(call: ast.Call, name: str) -> bool | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, bool) else None
    return None


def _nodes_in_scope(scope: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            if child is not scope and isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
            ):
                continue
            visit(child)

    for child in ast.iter_child_nodes(scope):
        if isinstance(
            child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            continue
        visit(child)
    return nodes


def _unsafe_pre_norm_encoders(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scopes = [
        tree,
        *(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ),
    ]
    unsafe: list[int] = []

    for scope in scopes:
        nodes = _nodes_in_scope(scope)
        layer_assignments: dict[str, list[tuple[int, bool]]] = {}
        for node in nodes:
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if _constructor_name(value) != "TransformerEncoderLayer":
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    layer_assignments.setdefault(target.id, []).append(
                        (node.lineno, _keyword_bool(value, "norm_first") is True)
                    )

        for node in nodes:
            if not isinstance(node, ast.Call) or _constructor_name(node) != "TransformerEncoder":
                continue
            if not node.args:
                continue

            layer_arg = node.args[0]
            is_pre_norm = False
            if isinstance(layer_arg, ast.Name):
                previous = [
                    assignment
                    for assignment in layer_assignments.get(layer_arg.id, [])
                    if assignment[0] < node.lineno
                ]
                if previous:
                    is_pre_norm = max(previous, key=lambda assignment: assignment[0])[1]
            elif _constructor_name(layer_arg) == "TransformerEncoderLayer":
                is_pre_norm = _keyword_bool(layer_arg, "norm_first") is True

            if is_pre_norm and _keyword_bool(node, "enable_nested_tensor") is not False:
                unsafe.append(node.lineno)

    return unsafe


def test_pre_norm_transformer_encoders_explicitly_disable_nested_tensors() -> None:
    unsafe = {
        str(path.relative_to(REPO_ROOT)): lines
        for path in (REPO_ROOT / "dlhub").rglob("*.py")
        if (lines := _unsafe_pre_norm_encoders(path))
    }

    assert unsafe == {}
