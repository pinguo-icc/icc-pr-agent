"""Batch splitter for splitting large file groups into smaller batches."""

from __future__ import annotations

import os

from src.models import Batch, FileGroup


class BatchSplitter:
    """组内二次分片器。"""

    @staticmethod
    def _sort_key(path: str) -> tuple[str, str]:
        """Sort by directory prefix then basename for locality."""
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        return (dirname, basename)

    @staticmethod
    def _stem(path: str) -> str:
        """Extract stem without extension: 'a/user_service.go' -> 'user_service'."""
        base = os.path.basename(path)
        return os.path.splitext(base)[0]

    def split(self, group: FileGroup, max_chunk_chars: int) -> list[Batch]:
        """将 FileGroup 按文件粒度分片。

        - 按目录前缀排序文件，使同目录文件相邻
        - 关联文件（如 user.proto + user_service.go）尽量同批
        - 单文件超限时独立成批
        """
        if not group.file_paths:
            return []

        # Pair paths with their diffs and sort by directory prefix
        items = list(zip(group.file_paths, group.file_diffs))
        items.sort(key=lambda x: self._sort_key(x[0]))

        batches: list[Batch] = []
        current_paths: list[str] = []
        current_diffs: list[str] = []
        current_chars = 0

        for path, diff_text in items:
            file_chars = len(diff_text)

            # Single file exceeds limit -> standalone batch
            if file_chars > max_chunk_chars and not current_paths:
                batches.append(Batch(
                    group_name=group.name,
                    batch_index=len(batches),
                    file_paths=[path],
                    diff_content=diff_text,
                    char_count=file_chars,
                ))
                continue

            # Adding this file would exceed limit -> flush current batch
            if current_chars + file_chars > max_chunk_chars and current_paths:
                batches.append(Batch(
                    group_name=group.name,
                    batch_index=len(batches),
                    file_paths=list(current_paths),
                    diff_content="".join(current_diffs),
                    char_count=current_chars,
                ))
                current_paths = []
                current_diffs = []
                current_chars = 0

            # Single file exceeds limit after flush -> standalone
            if file_chars > max_chunk_chars:
                batches.append(Batch(
                    group_name=group.name,
                    batch_index=len(batches),
                    file_paths=[path],
                    diff_content=diff_text,
                    char_count=file_chars,
                ))
                continue

            current_paths.append(path)
            current_diffs.append(diff_text)
            current_chars += file_chars

        # Flush remaining
        if current_paths:
            batches.append(Batch(
                group_name=group.name,
                batch_index=len(batches),
                file_paths=list(current_paths),
                diff_content="".join(current_diffs),
                char_count=current_chars,
            ))

        return batches
