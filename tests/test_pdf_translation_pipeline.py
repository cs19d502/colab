from __future__ import annotations

"""Tests for the Docling-based PDF translation pipeline."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from pdf_translation_pipeline import (
    ComponentType,
    DoclingExtractor,
    DocumentComponent,
    PageLayout,
    translate_pdf,
)


def _sample_brochure_doc() -> dict:
    return {
        "pages": [
            {
                "width": 600,
                "height": 780,
                "background_color": [0.9, 0.95, 1.0],
                "elements": [
                    {
                        "type": "header",
                        "text": "Industrial Sensor Brochure",
                        "bbox": [48, 48, 552, 90],
                        "font_size": 26,
                    },
                    {
                        "type": "text",
                        "text": "Engineered for harsh factory environments.",
                        "bbox": [48, 100, 552, 140],
                        "font_size": 12,
                    },
                    {
                        "type": "table",
                        "cells": [
                            ["Feature", "Specification"],
                            ["Ingress Protection", "IP67"],
                        ],
                        "bbox": [48, 160, 360, 240],
                    },
                    {
                        "type": "image",
                        "image": "inline-image-placeholder",
                        "bbox": [380, 160, 540, 320],
                    },
                ],
            }
        ]
    }


def _write_fixture(pdf_path: Path) -> Path:
    pdf_path.write_bytes(b"placeholder brochure pdf")
    json_path = pdf_path.with_suffix(".docling.json")
    json_path.write_text(json.dumps(_sample_brochure_doc()), encoding="utf-8")
    return json_path


def test_extractor_loads_components_from_json(tmp_path: Path) -> None:
    pdf_path = tmp_path / "brochure.pdf"
    _write_fixture(pdf_path)

    extractor = DoclingExtractor()
    components, layouts = extractor.extract(pdf_path)

    assert set(layouts.keys()) == {1}
    layout = layouts[1]
    assert isinstance(layout, PageLayout)
    assert layout.width == pytest.approx(600)
    assert layout.height == pytest.approx(780)
    assert layout.background_color == pytest.approx((0.9, 0.95, 1.0))

    types = [component.component_type for component in components]
    assert types.count(ComponentType.HEADER) == 1
    assert types.count(ComponentType.TEXT) == 1
    assert types.count(ComponentType.TABLE) == 1
    assert types.count(ComponentType.IMAGE) == 1


class DummyTranslator:
    def translate_component(self, component: DocumentComponent) -> DocumentComponent:
        if component.component_type == ComponentType.TABLE:
            table = component.content or []
            translated = []
            for row in table:
                if isinstance(row, list):
                    translated.append([f"JA:{cell}" for cell in row])
                else:
                    translated.append([f"JA:{row}"])
            return component.clone_with_content(translated)
        if component.component_type in {ComponentType.TEXT, ComponentType.HEADER}:
            text = component.content or ""
            return component.clone_with_content(f"JA:{text}")
        return component


class CaptureRenderer:
    def __init__(self) -> None:
        self.components = []
        self.page_layouts = {}
        self.output_path: Path | None = None

    def render(self, components, page_layouts, output_path: Path) -> None:
        self.components = list(components)
        self.page_layouts = dict(page_layouts)
        self.output_path = output_path
        output_path.write_text("stub pdf", encoding="utf-8")


def test_translate_pdf_uses_injected_dependencies(tmp_path: Path) -> None:
    pdf_path = tmp_path / "brochure.pdf"
    json_path = _write_fixture(pdf_path)
    output_pdf = tmp_path / "brochure-ja.pdf"

    extractor = DoclingExtractor()
    translator = DummyTranslator()
    renderer = CaptureRenderer()

    translate_pdf(
        input_pdf=pdf_path,
        output_pdf=output_pdf,
        language="Japanese",
        model="dummy-model",
        extractor=extractor,
        translator=translator,
        renderer=renderer,
        docling_json=json_path,
    )

    assert renderer.output_path == output_pdf
    assert output_pdf.exists()

    header = next(
        component for component in renderer.components if component.component_type == ComponentType.HEADER
    )
    assert str(header.content).startswith("JA:")

    table = next(
        component for component in renderer.components if component.component_type == ComponentType.TABLE
    )
    assert table.content[0][0] == "JA:Feature"

    layout = renderer.page_layouts[1]
    assert layout.background_color == pytest.approx((0.9, 0.95, 1.0))
