"""Execute report.ipynb in the current project venv and export its HTML companion."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "report.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    runtime = ROOT / "results/notebook-verification"
    kernel_root = runtime / "kernels"
    kernel = kernel_root / "project-venv"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python (project venv)", "language": "python"}), encoding="utf-8")
    manager = KernelManager(kernel_name="project-venv", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client = NotebookClient(notebook, km=manager, timeout=120, allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}})
    started = time.perf_counter()
    try:
        client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    notebook.metadata.kernelspec = {"display_name": "Python (project venv)", "language": "python", "name": "python3"}
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    figures = []
    error_count = 0
    for i, cell in enumerate(notebook.cells):
        for j, output in enumerate(cell.get("outputs", [])):
            error_count += output.output_type == "error"
            if "image/png" in output.get("data", {}):
                figure = runtime / f"cell-{i:03d}-output-{j:02d}.png"
                figure.write_bytes(base64.b64decode(output.data["image/png"]))
                figures.append(figure.relative_to(ROOT).as_posix())
    assert error_count == 0
    exporter = HTMLExporter(template_name="lab", embed_images=True)
    html, _ = exporter.from_notebook_node(notebook)
    css = """<style>
    body { background:#f4f7fa; }
    .jp-Notebook { max-width:1120px; margin:24px auto; background:white; padding:34px 44px; }
    .jp-RenderedHTMLCommon { font-family:system-ui,Segoe UI,sans-serif; line-height:1.7; color:#23384c; }
    .jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2 { color:#163d59; }
    .jp-RenderedHTMLCommon h2 { margin-top:2em; border-bottom:2px solid #e6eff2; padding-bottom:.45em; }
    .jp-RenderedHTMLCommon blockquote { border-left:4px solid #168c82; background:#eef8f5; padding:12px 18px; }
    .jp-RenderedHTMLCommon table { font-size:.94em; }
    .jp-OutputArea-output img { max-width:100%; height:auto; }
    @media print { .jp-Notebook {max-width:none; margin:0; padding:0;} h2 {break-after:avoid;} img {break-inside:avoid;} }
    </style>"""
    (ROOT / "report.html").write_text(html.replace("</head>", css + "</head>"), encoding="utf-8")
    facts = dict(status="passed", python=sys.executable, seconds=time.perf_counter()-started,
                 cells=len(notebook.cells), code_cells=sum(c.cell_type == "code" for c in notebook.cells),
                 markdown_cells=sum(c.cell_type == "markdown" for c in notebook.cells),
                 errors=error_count, figures=figures,
                 sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["report.ipynb", "report.html"]})
    (runtime / "execution.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in facts.items() if k not in ["figures", "sha256"]}, indent=2))
    print(f"Saved report.ipynb and report.html with {len(figures)} embedded figures.")


if __name__ == "__main__":
    main()
