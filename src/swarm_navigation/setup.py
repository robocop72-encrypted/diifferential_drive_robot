from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'swarm_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tushar',
    maintainer_email='tushar@todo.com',
    description='Layer 1 sensor fusion',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sensor_fusion = swarm_navigation.sensor_fusion:main',
            'obstacle_classifier = swarm_navigation.obstacle_classifier:main',
        ],
    },
)
