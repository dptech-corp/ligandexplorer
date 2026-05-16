from setuptools import setup, find_packages
import os
import glob

def wrapped_setup(scm=None):
    setup(
        name = 'Ligandexplorer',
        use_scm_version=scm,
        setup_requires=['setuptools_scm'],
        author='DP Tech',
        author_email='liyq@dp.tech',
        description=('Find ligand from strcuture and structured data'),
        license='MIT',
        install_requires=[
                          'rarfile',
                          'networkx',
                          'biopython',
                          'numpy',
                          'torch',
                          'torch_geometric',
                          'scipy',
                          ],
        extras_require={
            'lgbm': [
                'lightGBM',
                'scikit-learn>=1.5',
                'rdkit>=2024.3',
                'pandas',
                'joblib',
            ],
        },
        packages=find_packages(),
        package_data={'ligandexplorer': ["model/*"]},
        zip_safe = False,
        entry_points={'console_scripts': [
            'ligandexplorer = ligandexplorer.workflow:main'
        ]},
        include_package_data=True
    )

try:
    main_folder_name = 'ligandexplorer'
    wrapped_setup(scm={'write_to': os.path.join(main_folder_name, "_version.py")})
except Exception as e:
    print(e)
    wrapped_setup(scm=None)
