from setuptools import setup, find_packages

with open("readme.md", 'r') as fh:
    long_description = fh.read()
    
__version__ = '0.0.2'
URL = None
install_requires = [
    "scipy",
    "protobuf",
    "networkx",
    "libwon",
    "snap-stanford",
    'igraph',
    'openai==0.28',
    'scikit-learn',
]

setup(
    name='llm4dyg',
    version=__version__,
    author='wondergo',
    author_email='wondergo2017@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = URL,
    python_requires='>=3.9',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)
