

from setuptools import setup, find_packages

setup(
    # Basic package information:
    name    ='cgem',  
    version ='0.1.0', 
    packages=find_packages(),  # Automatically find packages in the directory

    # Dependencies:
    install_requires=[
        'numpy>=1.1.1',  
        'pandas',
        'pandas_ta',
        'scipy',
        'scikit-learn',
        'sympy',
        'statsmodels',
        'pygam',
        'xgboost' 
    ],

    # Metadata for PyPI:
    author          ='James A. Rolfsen',
    author_email    ='james.rolfsen@think.dev', 
	description     ='CGEM: Collaborative Generalized Effects Modeling',
	url             ='https://github.com/jrolf/cgem',    # Link to your repo
    license         ='MIT',
    
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',  # If your README is in markdown

    # More classifiers: https://pypi.org/classifiers/
    classifiers=[
        'Programming Language :: Python :: 3.7', 
        'License :: OSI Approved :: MIT License',  # Ensure it matches the LICENSE
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    

)





