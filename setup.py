from setuptools import setup, find_packages

setup(
    name='unslothlora',
    version='0.1',
    packages=find_packages(),
    install_requires=[], # xformers is a huge pita so we're not going to install it here. Colab script is the best way to install it.
)
