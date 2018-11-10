from setuptools import setup, find_packages

setup(name='rivalgan',
      version='0.4',
      url='https://github.com/InsightDataCommunity/Rival-AI.NY-18C',
      author='Yves Greatti',
      author_email='yvgrotti@gmail.com',
      description='Generate business data to balance classes',
      packages=find_packages(exclude=['tests']),
      long_description=open('README.md').read(),
      zip_safe=False)
