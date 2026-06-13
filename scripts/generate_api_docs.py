from dataclasses import dataclass
from pathlib import Path

from griffe import Class, Function, load

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "content" / "docs" / "api"
SEARCH_PATHS = [str(ROOT / "src")]


@dataclass(frozen=True)
class ApiObject:
    module: str
    name: str
    title: str | None = None


FUNCTIONAL_API = [
    ApiObject("greyhound.nn.functional", "cross_entropy"),
    ApiObject("greyhound.nn.functional", "autograd_loss_and_logits_grad"),
    ApiObject("greyhound.nn.functional", "chunked_linear_loss"),
    ApiObject("greyhound.nn.functional", "chunked_linear_cross_entropy"),
    ApiObject("greyhound.nn.functional", "causal_conv1d"),
    ApiObject("greyhound.nn.functional", "selective_log_softmax"),
]

MODULE_API = [
    ApiObject("greyhound.nn.causal_conv1d", "GreyhoundCausalConv1d"),
]


def normalize_doc_text(text: str) -> str:
    return text.strip().replace("``", "`")


def render_annotation(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).replace("typing.", "")


def render_default(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).replace("ExprUnaryOp(operator='-', value='100')", "-100")


def indent_description(text: str) -> str:
    return normalize_doc_text(text).replace("\n", "\n  ")


def split_docstring(obj: Function | Class) -> dict[str, object]:
    sections: dict[str, object] = {"text": "", "parameters": [], "returns": []}
    if obj.docstring is None:
        return sections

    for section in obj.docstring.parse("google"):
        kind = section.kind.value
        if kind == "text":
            sections["text"] = normalize_doc_text(section.value)
        elif kind == "parameters":
            sections["parameters"] = section.value
        elif kind == "returns":
            sections["returns"] = section.value
    return sections


def render_signature(obj: Function | Class, api_name: str) -> str:
    if isinstance(obj, Function):
        return str(obj.signature())

    init = obj.members.get("__init__")
    if not isinstance(init, Function):
        return f"{api_name}()"

    parameters = []
    for parameter in init.parameters:
        if parameter.name == "self":
            continue
        rendered = parameter.name
        annotation = render_annotation(parameter.annotation)
        default = render_default(parameter.default)
        if annotation:
            rendered += f": {annotation}"
        if default:
            rendered += f" = {default}"
        parameters.append(rendered)
    return f"{api_name}({', '.join(parameters)})"


def render_parameters(parameters: object) -> list[str]:
    if not parameters:
        return []

    lines = ["**Parameters**"]
    for parameter in parameters:
        annotation = render_annotation(parameter.annotation)
        default = render_default(parameter.value)
        suffix_parts = []
        if annotation:
            suffix_parts.append(annotation)
        if default:
            suffix_parts.append(f"default: `{default}`")
        suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
        description = indent_description(parameter.description)
        lines.append(f"- `{parameter.name}`{suffix}: {description}")
    return lines


def render_returns(returns: object) -> list[str]:
    if not returns:
        return []

    lines = ["**Returns**"]
    returns = list(returns)
    annotations = {render_annotation(returned.annotation) for returned in returns}
    if len(annotations) == 1:
        annotation = annotations.pop()
        if annotation:
            lines.extend(["", f"`{annotation}`"])
        descriptions = [normalize_doc_text(returned.description) for returned in returns]
        lines.extend(["", "\n".join(descriptions)])
        return lines

    for returned in returns:
        annotation = render_annotation(returned.annotation)
        suffix = f" (`{annotation}`)" if annotation else ""
        description = indent_description(returned.description)
        lines.append(f"-{suffix}: {description}")
    return lines


def render_methods(obj: Class) -> list[str]:
    lines: list[str] = []
    for method_name in ("forward", "reset_parameters"):
        method = obj.members.get(method_name)
        if not isinstance(method, Function) or method.docstring is None:
            continue
        sections = split_docstring(method)
        lines.extend(["", f"#### `{method_name}`", "", "```python", str(method.signature()), "```"])
        text = sections["text"]
        if text:
            lines.extend(["", str(text)])
        params = render_parameters(sections["parameters"])
        if params:
            lines.extend(["", *params])
        returns = render_returns(sections["returns"])
        if returns:
            lines.extend(["", *returns])
    return lines


def render_object(api_object: ApiObject) -> str:
    module = load(api_object.module, search_paths=SEARCH_PATHS)
    obj = module[api_object.name]
    if not isinstance(obj, Function | Class):
        raise TypeError(f"{api_object.module}.{api_object.name} is not documentable")

    title = api_object.title or api_object.name
    qualified_name = f"{api_object.module}.{api_object.name}"
    sections = split_docstring(obj)

    lines = [
        f"### `{title}`",
        "",
        f"`{qualified_name}`",
        "",
        "```python",
        render_signature(obj, api_object.name),
        "```",
    ]

    text = sections["text"]
    if text:
        lines.extend(["", str(text)])

    params = render_parameters(sections["parameters"])
    if params:
        lines.extend(["", *params])

    returns = render_returns(sections["returns"])
    if returns:
        lines.extend(["", *returns])

    if isinstance(obj, Class):
        lines.extend(render_methods(obj))

    return "\n".join(lines)


def render_page(title: str, description: str, objects: list[ApiObject]) -> str:
    body = "\n\n".join(render_object(obj) for obj in objects)
    return "\n".join(
        [
            "---",
            f"title: {title}",
            f"description: {description}",
            "---",
            "",
            "Generated from Python docstrings by `scripts/generate_api_docs.py`.",
            "",
            body,
            "",
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "functional.md").write_text(
        render_page(
            "Functional API",
            "Generated reference for greyhound.nn.functional.",
            FUNCTIONAL_API,
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "modules.md").write_text(
        render_page(
            "Module Wrappers",
            "Generated reference for greyhound.nn module wrappers.",
            MODULE_API,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
