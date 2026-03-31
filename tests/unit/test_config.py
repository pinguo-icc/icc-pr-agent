"""Unit tests for src/config.py."""

import yaml

from src.config import Config

# All tests use a non-existent dotenv path to avoid loading the real .env file.
_NO_DOTENV = "/tmp/__nonexistent__.env"


def _write_yaml(tmp_path, data: dict) -> str:
    """Write a YAML config file and return its path."""
    yaml_file = tmp_path / "pr-review.yaml"
    yaml_file.write_text(yaml.dump(data, default_flow_style=False))
    return str(yaml_file)


class TestConfigFromEnv:
    def test_defaults(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.github_token == ""
        assert cfg.gitlab_url == "https://gitlab.com"
        assert cfg.llm_model == "gpt-4"
        assert cfg.log_level == "INFO"
        assert cfg.review_storage_dir.endswith(".pr_reviews")
        assert cfg.pr_review_exclude == []
        assert cfg.skills_dir == ""

    def test_skills_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SKILLS_DIR", "/custom/skills")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.skills_dir == "/custom/skills"

    def test_loads_tokens(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GITHUB_TOKEN", "gh_tok")
        monkeypatch.setenv("GITLAB_TOKEN", "gl_tok")
        monkeypatch.setenv("CODEUP_TOKEN", "cu_tok")
        monkeypatch.setenv("LLM_API_KEY", "sk-key")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.github_token == "gh_tok"
        assert cfg.gitlab_token == "gl_tok"
        assert cfg.codeup_token == "cu_tok"
        assert cfg.llm_api_key == "sk-key"

    def test_loads_custom_values(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GITLAB_URL", "https://git.example.com")
        monkeypatch.setenv("LLM_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("REVIEW_STORAGE_DIR", "/tmp/reviews")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.gitlab_url == "https://git.example.com"
        assert cfg.llm_model == "gpt-3.5-turbo"
        assert cfg.llm_base_url == "https://api.example.com"
        assert cfg.log_level == "DEBUG"
        assert cfg.review_storage_dir == "/tmp/reviews"

    def test_parses_exclude_patterns(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PR_REVIEW_EXCLUDE", "*.lock,*.png, docs/**")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.pr_review_exclude == ["*.lock", "*.png", "docs/**"]

    def test_empty_exclude_patterns(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.pr_review_exclude == []

    def test_webhook_secrets(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WEBHOOK_SECRET_GITHUB", "wh_gh")
        monkeypatch.setenv("WEBHOOK_SECRET_GITLAB", "wh_gl")
        monkeypatch.setenv("WEBHOOK_SECRET_CODEUP", "wh_cu")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.webhook_secret_github == "wh_gh"
        assert cfg.webhook_secret_gitlab == "wh_gl"
        assert cfg.webhook_secret_codeup == "wh_cu"

    def test_codeup_org_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CODEUP_ORG_ID", "org123")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.codeup_org_id == "org123"

    def test_loads_from_dotenv_file(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("GITHUB_TOKEN=from_file\nLLM_MODEL=claude\n")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=str(dotenv), yaml_path=yaml_path)
        assert cfg.github_token == "from_file"
        assert cfg.llm_model == "claude"


class TestConfigNewFieldDefaults:
    """Test default values for new sub-agent review fields."""

    def test_default_file_groups_is_none(self, tmp_path):
        """Validates: Requirement 2.1 — file_groups defaults to None."""
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.file_groups is None

    def test_default_max_chunk_chars_is_none(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_chunk_chars is None

    def test_default_max_issues(self, tmp_path):
        """Validates: Requirement 6.2 — max_issues defaults to 10."""
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 10

    def test_default_max_concurrency(self, tmp_path):
        """Validates: Requirement 10.2 — max_concurrency defaults to 3."""
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_concurrency == 3

    def test_default_review_timeout(self, tmp_path):
        """Validates: Requirement 12.2 — review_timeout defaults to 300."""
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.review_timeout == 300


class TestConfigFileGroupsFromYaml:
    """Test file_groups loading from YAML config.

    Validates: Requirement 2.1
    """

    def test_loads_file_groups_from_yaml(self, tmp_path):
        yaml_data = {
            "file_groups": {
                "backend": ["*.go", "*.proto"],
                "frontend": ["*.ts", "*.tsx", "*.vue"],
            }
        }
        yaml_path = _write_yaml(tmp_path, yaml_data)
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.file_groups == {
            "backend": ["*.go", "*.proto"],
            "frontend": ["*.ts", "*.tsx", "*.vue"],
        }

    def test_invalid_file_groups_type_ignored(self, tmp_path):
        """Non-dict file_groups should be ignored with a warning."""
        yaml_data = {"file_groups": "not-a-dict"}
        yaml_path = _write_yaml(tmp_path, yaml_data)
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.file_groups is None

    def test_file_groups_not_present_in_yaml(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_issues": 5})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.file_groups is None


class TestConfigYamlValues:
    """Test loading max_issues, max_concurrency, review_timeout from YAML."""

    def test_max_issues_from_yaml(self, tmp_path):
        """Validates: Requirement 6.2"""
        yaml_path = _write_yaml(tmp_path, {"max_issues": 20})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 20

    def test_max_concurrency_from_yaml(self, tmp_path):
        """Validates: Requirement 10.2"""
        yaml_path = _write_yaml(tmp_path, {"max_concurrency": 8})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_concurrency == 8

    def test_review_timeout_from_yaml(self, tmp_path):
        """Validates: Requirement 12.2"""
        yaml_path = _write_yaml(tmp_path, {"review_timeout": 600})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.review_timeout == 600

    def test_max_chunk_chars_from_yaml(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_chunk_chars": 25000})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_chunk_chars == 25000


class TestConfigEnvOverridesYaml:
    """Test that environment variables override YAML config values.

    Validates: Requirements 6.2, 10.2, 12.2
    """

    def test_env_overrides_max_issues(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_REVIEW_ISSUES", "25")
        yaml_path = _write_yaml(tmp_path, {"max_issues": 15})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 25

    def test_env_overrides_max_concurrency(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_REVIEW_CONCURRENCY", "6")
        yaml_path = _write_yaml(tmp_path, {"max_concurrency": 2})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_concurrency == 6

    def test_env_overrides_review_timeout(self, monkeypatch, tmp_path):
        monkeypatch.setenv("REVIEW_TIMEOUT", "120")
        yaml_path = _write_yaml(tmp_path, {"review_timeout": 600})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.review_timeout == 120

    def test_env_overrides_yaml_even_when_yaml_absent(self, monkeypatch, tmp_path):
        """Env var should work even without YAML config."""
        monkeypatch.setenv("MAX_REVIEW_ISSUES", "7")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 7


class TestConfigInvalidValuesFallback:
    """Test that invalid values fall back to defaults.

    Validates: Requirements 6.4, 10.4, 12.5
    """

    def test_invalid_max_issues_in_yaml_falls_back(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_issues": "abc"})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 10

    def test_zero_max_issues_in_yaml_falls_back(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_issues": 0})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 10

    def test_negative_max_issues_in_yaml_falls_back(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_issues": -5})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 10

    def test_invalid_max_concurrency_in_yaml_falls_back(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_concurrency": "not_a_number"})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_concurrency == 3

    def test_zero_max_concurrency_in_yaml_falls_back(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_concurrency": 0})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_concurrency == 3

    def test_invalid_review_timeout_in_yaml_falls_back(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"review_timeout": -100})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.review_timeout == 300

    def test_invalid_max_issues_in_env_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_REVIEW_ISSUES", "not_int")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_issues == 10

    def test_invalid_max_concurrency_in_env_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_REVIEW_CONCURRENCY", "0")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_concurrency == 3

    def test_invalid_review_timeout_in_env_falls_back(self, monkeypatch, tmp_path):
        monkeypatch.setenv("REVIEW_TIMEOUT", "-1")
        yaml_path = _write_yaml(tmp_path, {})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.review_timeout == 300

    def test_invalid_max_chunk_chars_in_yaml_ignored(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_chunk_chars": "bad"})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_chunk_chars is None

    def test_zero_max_chunk_chars_in_yaml_ignored(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_chunk_chars": 0})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_chunk_chars is None

    def test_negative_max_chunk_chars_in_yaml_ignored(self, tmp_path):
        yaml_path = _write_yaml(tmp_path, {"max_chunk_chars": -500})
        cfg = Config.from_env(dotenv_path=_NO_DOTENV, yaml_path=yaml_path)
        assert cfg.max_chunk_chars is None
