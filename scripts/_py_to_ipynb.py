"""
Converts each train_*.py / 0X_*.py script into a Jupyter notebook.
Splits the module docstring into a markdown title cell, and the remaining
code into cells at blank-line boundaries (each blank-line-separated block
in the original script becomes its own cell).
"""
import ast
import nbformat as nbf
import sys
import os

def script_to_notebook(py_path, ipynb_path):
    src = open(py_path).read()
    tree = ast.parse(src)
    docstring = ast.get_docstring(tree)

    # strip the module docstring text out of the source to avoid duplicating it
    lines = src.splitlines()
    body_start = 0
    if docstring is not None:
        # find end of the triple-quoted docstring block
        in_str = False
        quote = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not in_str and (stripped.startswith('"""') or stripped.startswith("'''")):
                quote = stripped[:3]
                in_str = True
                if stripped.count(quote) >= 2 and len(stripped) >= 6:
                    body_start = i + 1
                    break
                continue
            if in_str and quote in line:
                body_start = i + 1
                break

    code_lines = lines[body_start:]
    code = "\n".join(code_lines)

    # split into cells on blank lines (collapse multiple blanks first), but never
    # split in the middle of an indented block (e.g. inside a for/if/def) --
    # only treat a blank line as a cell boundary if the next non-blank line is
    # at zero indentation (top-level code).
    code_split_lines = code.splitlines()
    blocks = []
    current = []
    i = 0
    n_lines = len(code_split_lines)
    while i < n_lines:
        line = code_split_lines[i]
        if line.strip() == "":
            # look ahead to the next non-blank line
            j = i
            while j < n_lines and code_split_lines[j].strip() == "":
                j += 1
            next_is_top_level = (j >= n_lines) or (len(code_split_lines[j]) - len(code_split_lines[j].lstrip()) == 0)
            if next_is_top_level and current:
                blocks.append("\n".join(current))
                current = []
            # blank lines are dropped either way (not appended to current)
            i = j
            continue
        current.append(line)
        i += 1
    if current:
        blocks.append("\n".join(current))
    blocks = [b for b in blocks if b.strip()]

    nb = nbf.v4.new_notebook()
    cells = []
    if docstring:
        title = docstring.strip().splitlines()[0]
        rest = docstring.strip().splitlines()[1:]
        md = f"# {title}\n\n" + "\n".join(rest)
        cells.append(nbf.v4.new_markdown_cell(md))
    for b in blocks:
        cells.append(nbf.v4.new_code_cell(b))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    with open(ipynb_path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {ipynb_path} ({len(cells)} cells)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        "00_extract_data.py",
        "01_train_fo_flux_longterm.py",
        "02_train_fo_flux_shortterm.py",
        "03_train_ro_rejection.py",
        "04_train_toc_rejection.py",
        "05_improved_fo_flux_longterm.py",
        "06_improved_fo_flux_shortterm.py",
        "07_improved_ro_rejection.py",
        "08_improved_toc_rejection.py",
    ]
    out_dir = os.path.abspath(os.path.join(here, ".."))
    for s in scripts:
        py_path = os.path.join(here, s)
        ipynb_path = os.path.join(out_dir, s.replace(".py", ".ipynb"))
        script_to_notebook(py_path, ipynb_path)
