"""Convert ladder logic from GX Works3 exported PDF to ST (Structured Text).

This script extracts text from PDF files and attempts to convert ladder
instruction list statements to ST syntax compatible with nvTOOL 4.

Usage:
    python pdf_to_st.py input1.pdf input2.pdf -o output.st

The conversion is heuristic and only supports a limited subset of ladder
instructions (LD, AND, OR, OUT, SET, RST, etc.). The PDF is expected to
contain an instruction list representation rather than graphical ladder
diagrams.
"""

import argparse
from pathlib import Path
from typing import List

try:
    import pdfplumber
except ImportError:  # pragma: no cover - pdfplumber may not be installed
    pdfplumber = None


def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using pdfplumber."""
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is required. Install it with `pip install pdfplumber`."
        )

    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def split_rungs(text: str) -> List[List[str]]:
    """Split extracted text into rungs of instruction list lines."""
    rungs: List[List[str]] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                rungs.append(current)
                current = []
            continue
        if line.lower().startswith(("rung", "network")):
            if current:
                rungs.append(current)
                current = []
            continue
        current.append(line)
    if current:
        rungs.append(current)
    return rungs


def _parse_instruction(instr: str) -> (str, str):
    """Parse an instruction list line into opcode and operand."""
    parts = instr.split(maxsplit=1)
    if not parts:
        return "", ""
    opcode = parts[0].upper()
    operand = parts[1].strip() if len(parts) > 1 else ""
    return opcode, operand


def rung_to_expression(rung: List[str]):
    """Convert a rung to ST expression and coil assignment."""
    expr_parts: List[str] = []
    coil_assignment = None

    for instr in rung:
        op, operand = _parse_instruction(instr)
        if op in {"LD", "LDP", "LDN"}:
            if op == "LDN":
                expr_parts = [f"NOT {operand}"]
            else:
                expr_parts = [operand]
        elif op in {"AND", "ANDP", "ANDN"}:
            if op == "ANDN":
                expr_parts.append(f"AND NOT {operand}")
            else:
                expr_parts.append(f"AND {operand}")
        elif op in {"OR", "ORP", "ORN"}:
            if op == "ORN":
                expr_parts.append(f"OR NOT {operand}")
            else:
                expr_parts.append(f"OR {operand}")
        elif op in {"OUT", "SET", "RST", "OUTNOT"}:
            coil_assignment = (op, operand)
        # additional instructions could be handled here

    expr = " ".join(expr_parts)
    return expr, coil_assignment


def rung_to_st(rung: List[str], index: int) -> str:
    expr, coil_assign = rung_to_expression(rung)
    if not coil_assign:
        return f"(* Rung {index}: no coil found *)"

    opcode, coil = coil_assign
    st_lines = [f"(* Rung {index} *)"]
    if opcode == "OUT":
        st_lines.append(f"{coil} := {expr};")
    elif opcode == "OUTNOT":
        st_lines.append(f"{coil} := NOT ({expr});")
    elif opcode == "SET":
        st_lines.append(f"IF {expr} THEN {coil} := TRUE; END_IF;")
    elif opcode == "RST":
        st_lines.append(f"IF {expr} THEN {coil} := FALSE; END_IF;")
    else:
        st_lines.append(f"(* Unsupported coil opcode {opcode} *)")
    return "\n".join(st_lines)


def convert_text_to_st(text: str) -> str:
    rungs = split_rungs(text)
    st_lines: List[str] = []
    for i, rung in enumerate(rungs, 1):
        st = rung_to_st(rung, i)
        st_lines.append(st)
        st_lines.append("")
    return "\n".join(st_lines)


def convert_pdfs_to_st(pdf_paths: List[Path]) -> str:
    all_text = []
    for pdf in pdf_paths:
        text = extract_text(pdf)
        all_text.append(text)
    return convert_text_to_st("\n".join(all_text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GX Works3 PDF to ST")
    parser.add_argument("pdfs", nargs="+", type=Path, help="Input PDF files")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output.st"),
        help="Output ST file path"
    )
    args = parser.parse_args()

    st_code = convert_pdfs_to_st(args.pdfs)
    args.output.write_text(st_code, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
