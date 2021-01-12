from distutils.core import setup

setup(name='multilens',
      version=0.01,
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'sklearn'],
      py_modules=['multilens', 'multilens_utils'])
