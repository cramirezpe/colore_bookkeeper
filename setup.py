import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="colore_bookkeeper",
    version="0.1",
    author="",
    author_email="user@host.com",
    description="Tool to run CoLoRe and LyaCoLoRe scripts",
    long_description=long_description,
    long_description_content="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={
        "colore_bookkeeper": [
            "resources/*",
            "resources/*/*",
        ]
    },
    # include_package_data=True,
    entry_points={
        "console_scripts": [
            "colore_bookkeeper_run_colore = colore_bookkeeper.scripts.run_CoLoRe:main",
            "colore_bookkeeper_run_corrf = colore_bookkeeper.scripts.run_Corrf:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.12",
)
