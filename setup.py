from setuptools import setup, find_packages

setup(
    name='unslothlora',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch",
        "unsloth @ git+https://github.com/unslothai/unsloth.git",
        "xformers",
        "trl",
        "peft",
        "accelerate",
        "bitsandbytes"
    ],
)
