"""Conftest for pytest-rst."""

import ast
import logging
from pathlib import Path
from typing import Any, List

import docutils.core  # type: ignore
import docutils.nodes  # type: ignore
import pytest

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


RST_FILES_TO_TEST = [
    Path(__file__).parent.parent / "README.rst",
]


def extract_python_blocks(content) -> List[str]:
    """Extract all Python code blocks from the RST content.

    This function parses the provided RST content and extracts all the
    Python code blocks marked by `.. code-block:: python`
    or `.. code:: python`. The extracted code blocks are returned as a
    list of strings. For isntance, given an RST content with a
    Python code block:
        .. code-block:: python

            print("Hello, world!")

    This function would return: ["print('Hello, world!')"]

    Parameters
    ----------
    content : str
        The reStructuredText (RST) content to be parsed and searched for
        Python code blocks.

    Returns
    -------
    List[str]
        A list of strings, where each string is a Python code block extracted
        from the RST content.

    """
    document = docutils.core.publish_doctree(content)
    code_blocks = [
        node.astext()
        for node in document.traverse(docutils.nodes.literal_block)
        if "python" in node.get("classes", [])
    ]
    return code_blocks


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: Any) -> None:
    """Run tests (hook) on specified .rst files at pytest session start.

    This function reads through a list of predefined .rst files,
    extracts Python code blocks, and ensures that the code is syntactically
    valid and that all necessary imports work correctly.
    The function will scan each file listed in `RST_FILES_TO_TEST`.

    This function is invoked automatically by pytest when the session starts.
    No manual invocation is needed.

    Parameters
    ----------
    session : Any
        The pytest session object. This hook is automatically called by pytest
        at the start of the session. It is not used in this function but is
        required by the pytest hook mechanism.

    Raises
    ------
    pytest.fail
        Raised if there is a syntax error in any of the Python code blocks,
        or if there is an import failure for any of the modules used
        in the code blocks.

    """
    for rst_file in RST_FILES_TO_TEST:
        if rst_file.exists():
            logging.info(f"Testing Python code in {rst_file}.")
            with open(rst_file) as f:
                content = f.read()
            code_blocks = extract_python_blocks(content)

            for i, code_block in enumerate(code_blocks):
                try:
                    tree = ast.parse(code_block)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                try:
                                    __import__(name.name)
                                except ImportError as e:
                                    pytest.fail(
                                        f"Cannot import {name.name} in "
                                        f"{rst_file}: {str(e)}"
                                    )
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                try:
                                    __import__(node.module)  # noqa
                                except ImportError as e:
                                    pytest.fail(
                                        f"Cannot import {node.module} in "
                                        f"{rst_file}: {str(e)}."
                                    )
                            else:
                                pytest.fail(
                                    f"Module name is None in {rst_file} "
                                    "in ImportFrom statement."
                                )
                except SyntaxError as e:
                    pytest.fail(
                        "Invalid Python syntax in code block "
                        f"{i + 1} in {rst_file}: "
                        f"\n{code_block}\nError: {str(e)}"
                    )
        else:
            logging.info(f"File {rst_file} does not exist, skippin...")
