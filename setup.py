import sys

from pkg_resources import VersionConflict, require
from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

try:
    require("setuptools>=38.3")
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)


if __name__ == "__main__":

    setup(
        name="pyannote.audio",
        namespace_packages=["pyannote"],
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        packages=find_packages(),
        install_requires=requirements,
        description="Neural building blocks for speaker diarization",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Hervé Bredin",
        author_email="bredin@limsi.fr",
        url="https://github.com/pyannote/pyannote-audio",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Natural Language :: English",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Topic :: Scientific/Engineering",
        ],
    )
