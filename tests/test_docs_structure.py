from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _relative_markdown_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*.md"))


def test_bilingual_docs_trees_are_mirrored():
    en_root = PROJECT_ROOT / "docs" / "en"
    zh_root = PROJECT_ROOT / "docs" / "zh"

    assert en_root.is_dir()
    assert zh_root.is_dir()
    assert _relative_markdown_files(en_root) == _relative_markdown_files(zh_root)


def test_readme_links_to_bilingual_docs():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/en/README.md" in readme
    assert "docs/zh/README.md" in readme
