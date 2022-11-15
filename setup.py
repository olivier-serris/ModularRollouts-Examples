from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "modular_rollouts @ git+https://github.com/olivier-serris/ModularRollouts.git",
    "codetiming",
    "cmake",
    "dm-haiku",
    "chex>=0.1.5",
    "rlax",
    "observable",
    "wandb",
]

setup(
    name="rlax_tests",
    version="0.0.1",
    author="Olivier Serris",
    author_email="serris@isir.upmc.fr",
    description="Jax tests  ",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/olivier-serris/ModularRollouts-Examples.git",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={"console_scripts": ["hydra_app = hydra_app.main:main"]},
    python_requires=">=3.7",
)
