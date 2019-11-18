#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="aprl",
    version=0.2,
    description="Adversarial Policies for Reinforcement Learning",
    author="Adam Gleave, Michael Dennis, et al",
    author_email="adam@gleave.me",
    python_requires=">=3.7.0",
    url="https://github.com/HumanCompatibleAI/adversarial-policies",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"aprl": ["configs/multi/*.json", "configs/noise/*.json", "configs/rew/*.json"]},
    # We have some non-pip packages as requirements,
    # see requirements-build.txt and requirements.txt.
    install_requires=[],
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
