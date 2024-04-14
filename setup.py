from setuptools import setup

setup(
    name="multilens",
    version=0.01,
    packages=[""],
    package_dir={"": "."},
    install_requires=["numpy", "scipy", "scikit-learn"],
    py_modules=["multilens", "multilens_utils"],
)
