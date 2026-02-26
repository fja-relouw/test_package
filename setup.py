from setuptools import setup, find_packages

setup(
    name="abp-autoencoder",
    version="0.1.0",
    description="Pretrained LSTM autoencoder for encoding ABP beat segments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "abp_autoencoder": ["weights/*.pth"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "numpy>=1.21",
    ],
)
