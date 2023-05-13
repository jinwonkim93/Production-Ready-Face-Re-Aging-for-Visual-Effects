from setuptools import setup, find_packages

setup(
  name = 'fran',
  package_dir={"": "src"},
  packages=find_packages("src"),
  version = '0.0.1',
  license='MIT',
  description = 'fran',
  author = 'Jinwon Kim',
  author_email = 'code.eric22@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/jinwonkim93/Production-Ready-Face-Re-Aging-for-Visual-Effects',
  keywords = [
    'ReAging',
    'image'
  ],
  install_requires=[
    'Pillow',
    'opencv-python',
    'numpy',
    'torch',
    'torchvision'
  ],
)