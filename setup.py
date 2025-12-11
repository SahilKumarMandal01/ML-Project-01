"""
Setup configuration for the ML project package.

This script defines the package metadata and dependencies required
for installation. It also includes a helper utility to dynamically
parse dependencies from a requirements file.
"""

from setuptools import find_packages, setup
from typing import List

# Constant representing editable install notation used in requirements files.
HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    Parse a requirements file and return a cleaned list of dependencies.

    Parameters
    ----------
    file_path : str
        Path to the requirements file (e.g., 'requirements.txt').

    Returns
    -------
    List[str]
        A list of requirement strings with newline characters removed
        and excluding the editable installation directive '-e .'.
    """
    requirements: List[str] = []

    # Open the requirements file and process each line.
    with open(file_path, "r") as file_obj:
        raw_lines = file_obj.readlines()

        # Strip newline characters from each requirement.
        requirements = [line.strip() for line in raw_lines]

        # Remove editable install directive if present.
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# -------------------------------------------------------------------------
# Package setup configuration
# -------------------------------------------------------------------------
setup(
    name="mlproject",
    version="0.0.1",
    author="Sahil Kumar Mandal",
    author_email="thesahilmandal@gmail.com",
    packages=find_packages(),               # Automatically discover Python packages
    install_requires=get_requirements("requirements.txt"),  # Inject dependencies dynamically
)
