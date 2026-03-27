"""Property-based tests for FileGrouper.

Uses hypothesis to validate correctness properties from the design doc.
"""

import fnmatch

from hypothesis import given, settings
from hypothesis import strategies as st

from src.file_grouper import FileGrouper


# --- Strategies ---

_EXTENSIONS = [".go", ".proto", ".ts", ".tsx", ".js", ".jsx", ".vue",
               ".css", ".less", ".scss", ".yml", ".py", ".txt", ".md"]

_file_name = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True).map(
    lambda n: n + ".go"  # placeholder, overridden below
)


def _random_file_path():
    """Generate a random file path like 'src/pkg/file.go'."""
    dirs = st.lists(
        st.from_regex(r"[a-z][a-z0-9]{0,6}", fullmatch=True),
        min_size=0, max_size=3,
    )
    name = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)
    ext = st.sampled_from(_EXTENSIONS)
    return st.tuples(dirs, name, ext).map(
        lambda t: "/".join(t[0] + [t[1] + t[2]])
    )


def _random_glob_pattern():
    """Generate a simple glob pattern like '*.go' or 'Dockerfile*'."""
    return st.sampled_from([
        "*.go", "*.proto", "*.ts", "*.tsx", "*.js", "*.py",
        "*.yml", "*.css", "*.md", "*.txt", "Dockerfile*", "Makefile",
    ])


def _random_file_groups():
    """Generate an ordered dict of group_name -> [glob_patterns]."""
    group_entry = st.tuples(
        st.from_regex(r"[a-z]{2,8}", fullmatch=True),
        st.lists(_random_glob_pattern(), min_size=1, max_size=4),
    )
    return st.lists(group_entry, min_size=1, max_size=5).map(dict)


# --- Property 1: 文件分组首匹配优先 ---
# Feature: sub-agent-review, Property 1: 文件分组首匹配优先

@given(file_path=_random_file_path(), file_groups=_random_file_groups())
@settings(max_examples=100)
def test_property_1_first_match_wins(file_path, file_groups):
    """Each file is assigned to exactly one group (first match wins).

    Validates: Requirements 1.1, 1.2, 1.3
    """
    result = FileGrouper._match_group(file_path, file_groups)

    # Result must be a string
    assert isinstance(result, str)

    # If result is not 'default', verify it's the first matching group
    basename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    if result != "default":
        assert result in file_groups
        # Verify it's the FIRST matching group
        for gname, patterns in file_groups.items():
            for pat in patterns:
                try:
                    if fnmatch.fnmatch(basename, pat):
                        assert gname == result, (
                            f"Expected first match '{gname}' but got '{result}'"
                        )
                        return
                except Exception:
                    continue
    else:
        # Verify no group matched
        for gname, patterns in file_groups.items():
            for pat in patterns:
                try:
                    matched = fnmatch.fnmatch(basename, pat)
                except Exception:
                    continue
                if matched:
                    # If a group matched, result should not be 'default'
                    assert False, f"File matched group '{gname}' but got 'default'"


# --- Property 2: Diff 头部路径提取 ---
# Feature: sub-agent-review, Property 2: Diff 头部路径提取

@given(file_path=_random_file_path())
@settings(max_examples=100)
def test_property_2_diff_header_path_extraction(file_path):
    """Extracted path from diff header matches the b/ path.

    Validates: Requirements 1.5
    """
    header = f"diff --git a/{file_path} b/{file_path}"
    extracted = FileGrouper._extract_file_path(header)
    assert extracted == file_path


@given(file_path=_random_file_path())
@settings(max_examples=100)
def test_property_2_diff_header_with_rename(file_path):
    """Extraction works when a/ and b/ paths differ (rename)."""
    old_path = "old/" + file_path
    header = f"diff --git a/{old_path} b/{file_path}"
    extracted = FileGrouper._extract_file_path(header)
    assert extracted == file_path


# --- Property 3: Glob 模式匹配一致性 ---
# Feature: sub-agent-review, Property 3: Glob 模式匹配一致性

@given(file_path=_random_file_path(), pattern=_random_glob_pattern())
@settings(max_examples=100)
def test_property_3_glob_consistency(file_path, pattern):
    """FileGrouper matching is consistent with fnmatch.

    Validates: Requirements 2.3
    """
    basename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    expected = fnmatch.fnmatch(basename, pattern)

    # Use a single-group config to test
    groups = {"test_group": [pattern]}
    result = FileGrouper._match_group(file_path, groups)

    if expected:
        assert result == "test_group"
    else:
        assert result == "default"
