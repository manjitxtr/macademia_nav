from setuptools import find_packages, setup

package_name = 'macademia_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rmitaiil',
    maintainer_email='s4161048@student.rmit.edu.au',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                    'orchard_navigation = macademia_nav.orchard_navigation:main',
                    'mock_scan_publisher = macademia_nav.mock_scan_publisher:main',
                    'orchard_lane_navigator = macademia_nav.orchard_lane_navigator:main',
                    'nut_detector = macademia_nav.nut_detector:main',

        ],
    },
)
