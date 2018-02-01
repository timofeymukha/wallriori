from setuptools import setup
from setuptools import find_packages

setup(name='wallriori',
      version='0.0.1',
      description='A package for a-priori analysis of WMLES',
      url='https://github.com/timofeymukha/wallriori',
      author='Timofey Mukha',
      author_email='timofey.mukha@it.uu.se',
      packages=find_packages(),
      entry_points = {
          'console_scripts':[
                            ]
      },
      install_requires=[
                    'numpy',
                    'scipy',
                    'matplotlib',
                       ],
      license="GNU GPL 3",
      classifiers=[
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: MIT Licence"
      ],
      zip_safe=False)

