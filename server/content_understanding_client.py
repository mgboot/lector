"""
Minimal Azure AI Content Understanding client for OCR of scanned PDFs.

Based on the sample at:
https://github.com/Azure-Samples/azure-ai-content-understanding-python

Uses Entra ID (token-based) authentication.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

import requests


POLL_TIMEOUT_SECONDS = 600  # scanned PDFs can take several minutes


class ContentUnderstandingClient:

    def __init__(
        self,
        endpoint: str,
        api_version: str = "2025-11-01",
        token_provider: callable = None,
    ):
        if not token_provider:
            raise ValueError("token_provider must be provided.")
        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self._token_provider = token_provider
        self._logger = logging.getLogger(__name__)

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._token_provider()}"}

    def ensure_defaults(self, model_deployments: Dict[str, str]) -> None:
        """Set default model deployments, updating if any requested models are missing."""
        url = (
            f"{self._endpoint}/contentunderstanding/defaults"
            f"?api-version={self._api_version}"
        )
        # Check current defaults first
        resp = requests.get(url, headers=self._auth_headers())
        if resp.ok:
            current = resp.json().get("modelDeployments", {})
            if all(k in current for k in model_deployments):
                self._logger.info("Content Understanding defaults already set.")
                return

        headers = {"Content-Type": "application/merge-patch+json"}
        headers.update(self._auth_headers())
        body = {"modelDeployments": model_deployments}

        resp = requests.patch(url, headers=headers, json=body)
        if not resp.ok:
            raise RuntimeError(
                f"Failed to set Content Understanding defaults ({resp.status_code}): {resp.text}"
            )
        self._logger.info("Content Understanding defaults configured.")

    def begin_analyze_binary(self, analyzer_id: str, file_path: str) -> requests.Response:
        """POST a local file to the analyzeBinary endpoint."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()

        url = (
            f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}"
            f":analyzeBinary?api-version={self._api_version}"
        )
        headers = {"Content-Type": "application/octet-stream"}
        headers.update(self._auth_headers())

        response = requests.post(url=url, headers=headers, data=file_bytes)
        if not response.ok:
            raise RuntimeError(
                f"Content Understanding request failed ({response.status_code}): {response.text}"
            )
        return response

    def poll_result(
        self,
        response: requests.Response,
        timeout_seconds: int = POLL_TIMEOUT_SECONDS,
        polling_interval: int = 10,
    ) -> Dict[str, Any]:
        """Poll the operation-location header until the analysis completes."""
        operation_location = response.headers.get("operation-location", "")
        if not operation_location:
            raise ValueError("operation-location header missing from response.")

        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Content Understanding timed out after {timeout_seconds}s"
                )

            poll = requests.get(operation_location, headers=self._auth_headers())
            poll.raise_for_status()
            data = poll.json()
            status = data.get("status", "").lower()

            if status == "succeeded":
                self._logger.info(f"Analysis completed in {elapsed:.0f}s")
                return data
            elif status == "failed":
                raise RuntimeError(
                    f"Content Understanding analysis failed: {data}"
                )

            print(f"    OCR in progress ({elapsed:.0f}s elapsed)...")
            time.sleep(polling_interval)
