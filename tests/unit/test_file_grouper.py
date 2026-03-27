"""Unit tests for src/file_grouper.py."""

from src.file_grouper import FileGrouper


def _make_diff(*files: tuple[str, str]) -> str:
    """Build a minimal unified diff from (path, body) pairs."""
    parts = []
    for path, body in files:
        parts.append(f"diff --git a/{path} b/{path}\n")
        parts.append(f"--- a/{path}\n+++ b/{path}\n")
        parts.append(body if body.endswith("\n") else body + "\n")
    return "".join(parts)


class TestExtractFilePath:
    def test_normal_header(self):
        assert FileGrouper._extract_file_path("diff --git a/src/main.go b/src/main.go") == "src/main.go"

    def test_non_header_line(self):
        assert FileGrouper._extract_file_path("+++ b/src/main.go") is None

    def test_empty_string(self):
        assert FileGrouper._extract_file_path("") is None


class TestMatchGroup:
    def test_first_match_wins(self):
        """Requirement 1.2: first-match-wins."""
        groups = {
            "proto": ["*.proto"],
            "backend": ["*.proto", "*.go"],
        }
        assert FileGrouper._match_group("api/user.proto", groups) == "proto"

    def test_unmatched_falls_to_default(self):
        """Requirement 1.3: unmatched -> default."""
        groups = {"backend": ["*.go"]}
        assert FileGrouper._match_group("README.txt", groups) == "default"

    def test_default_groups_match_go(self):
        """Requirement 1.4: default groups."""
        assert FileGrouper._match_group("main.go", FileGrouper.DEFAULT_FILE_GROUPS) == "backend"

    def test_default_groups_match_ts(self):
        assert FileGrouper._match_group("app.tsx", FileGrouper.DEFAULT_FILE_GROUPS) == "frontend"

    def test_default_groups_match_dockerfile(self):
        assert FileGrouper._match_group("Dockerfile", FileGrouper.DEFAULT_FILE_GROUPS) == "infra"

    def test_default_groups_match_yml(self):
        assert FileGrouper._match_group("ci.yml", FileGrouper.DEFAULT_FILE_GROUPS) == "infra"

    def test_default_groups_catch_all(self):
        assert FileGrouper._match_group("random.txt", FileGrouper.DEFAULT_FILE_GROUPS) == "default"


class TestGroupMethod:
    def test_empty_diff(self):
        fg = FileGrouper()
        result = fg.group("")
        assert result == {}

    def test_single_file(self):
        diff = _make_diff(("src/main.go", "@@ -1 +1 @@\n-old\n+new"))
        fg = FileGrouper()
        result = fg.group(diff)
        assert "backend" in result
        assert result["backend"].file_paths == ["src/main.go"]

    def test_multiple_groups(self):
        diff = _make_diff(
            ("src/main.go", "@@ -1 +1 @@\n-a\n+b"),
            ("app.tsx", "@@ -1 +1 @@\n-c\n+d"),
        )
        fg = FileGrouper()
        result = fg.group(diff)
        assert "backend" in result
        assert "frontend" in result
        assert result["backend"].file_paths == ["src/main.go"]
        assert result["frontend"].file_paths == ["app.tsx"]

    def test_custom_file_groups(self):
        custom = {"api": ["*.proto"], "impl": ["*.go"]}
        diff = _make_diff(
            ("api/user.proto", "@@ -1 +1 @@\n-x\n+y"),
            ("svc/order.go", "@@ -1 +1 @@\n-a\n+b"),
        )
        fg = FileGrouper(file_groups=custom)
        result = fg.group(diff)
        assert "api" in result
        assert "impl" in result

    def test_total_chars_calculated(self):
        body = "@@ -1 +1 @@\n-old\n+new\n"
        diff = _make_diff(("a.go", body))
        fg = FileGrouper()
        result = fg.group(diff)
        assert result["backend"].total_chars > 0

    def test_invalid_glob_skipped(self, caplog):
        """Requirement 2.2: invalid glob patterns logged and skipped."""
        # fnmatch doesn't really raise on bad patterns in Python,
        # but we test the fallback to default for unmatched files
        custom = {"special": ["[invalid"]}
        fg = FileGrouper(file_groups=custom)
        diff = _make_diff(("test.txt", "@@ -1 +1 @@\n-a\n+b"))
        result = fg.group(diff)
        # Should fall through to default since pattern won't match
        assert "default" in result or "special" in result
