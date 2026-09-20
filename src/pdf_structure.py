import re


def extract_paper_titles(text):
    titles = []
    seen = set()

    patterns = [
        r'"([^"\n]+)"',
        r"\u201c([^\u201d\n]+)\u201d",
        r"(?im)^\s*Title:\s*(.+?)\s*$",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text or ""):
            title = " ".join(match.strip().split())
            if title and title not in seen:
                seen.add(title)
                titles.append(title)

    return titles


def extract_literature_entries(text):
    content = text or ""
    lines = content.splitlines()

    heading_pattern = re.compile(
        r"^\s*(\d+(?:\.\d+)*)?\.?\s*(literature survey|literature review|related work)\s*:?\s*$",
        re.IGNORECASE,
    )
    numbered_entry_pattern = re.compile(
        r"^\s*(\[\d+\]|\d+[\.\)])\s+.+"
    )
    generic_heading_pattern = re.compile(
        r"^\s*(chapter\s+\d+|references|appendix|conclusion|results|discussion|methodology|introduction)\s*$",
        re.IGNORECASE,
    )

    def normalize_entry(parts):
        entry = " ".join(part.strip() for part in parts if part.strip())
        entry = re.sub(r"\s+", " ", entry).strip()
        return entry

    def is_noisy_entry(entry):
        if len(entry) < 20:
            return True
        if len(entry.split()) < 4:
            return True
        if not re.search(r"[A-Za-z]", entry):
            return True
        return False

    section_start = None
    for index, line in enumerate(lines):
        if heading_pattern.match(line.strip()):
            section_start = index
            section_preview = "\n".join(lines[index:])[:5000]
            print(f"Literature section header: {line.strip()}")
            print(section_preview)
            print("=" * 60)
            break

    if section_start is None:
        return []

    section_lines = []
    for line in lines[section_start + 1:]:
        stripped = line.strip()

        if generic_heading_pattern.match(stripped):
            break

        if re.match(r"^\s*\d+(?:\.\d+){1,}\s+.+", stripped):
            break

        section_lines.append(line)

    entries = []
    current_entry = []

    for line in section_lines:
        stripped = line.strip()

        if not stripped:
            continue

        if numbered_entry_pattern.match(stripped):
            if current_entry:
                entry = normalize_entry(current_entry)
                if not is_noisy_entry(entry):
                    entries.append({"raw": entry})
            current_entry = [stripped]
            continue

        if current_entry:
            current_entry.append(stripped)

    if current_entry:
        entry = normalize_entry(current_entry)
        if not is_noisy_entry(entry):
            entries.append({"raw": entry})

    return entries
