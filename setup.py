
from setuptools import setup, find_packages
from pathlib import Path

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / 'GPdoemd' / '__version__.py'
with version_path.open() as f:
    exec(f.read(), about)

setup(
    name='GPdoemd',
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    version=about['__version__'],
    packages=find_packages(exclude=['tests','docs']),
    requires=['pyomo', 'numpy', 'mpmath'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov'],
)