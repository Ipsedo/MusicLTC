# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="music_ltc",
    version="1.0.0",
    author="Samuel Berrien",
    packages=find_packages(include=["music_ltc", "music_ltc.*"]),
)
