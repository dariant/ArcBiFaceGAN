# ArcBiFaceGAN: Generating Bimodal Privacy-Preserving Data for Face Recognition


<!-- ![Teaser image](./docs/ArcBiFaceGAN_pipeline.png) -->

**ArcBiFaceGAN: Generating Bimodal Privacy-Preserving Data for Face Recognition**<br>
Darian Tomašević, Fadi Boutros, Naser Damer, Peter Peer, Vitomir Štruc<br>
LINK_TO_BE_ADDED<br>

Abstract : *The performance of state-of-the-art face recognition systems depends crucially on the availability of
large-scale training datasets. However, increasing privacy concerns nowadays accompany the collection
and distribution of biometric data, which already resulted in the retraction of valuable face recognition
datasets. The use of synthetic data represents a potential solution, however, the generation of privacy-preserving facial images useful for training recognition models is still an open problem. Generative methods
also remain bound to the visible spectrum, despite the benefits that multispectral data can provide. To
address these issues, we present a novel identity-conditioned generative framework capable of producing
large-scale recognition datasets of visible and near-infrared privacy-preserving face images. The framework
relies on a novel identity-conditioned dual-branch style-based generative adversarial network to enable the
synthesis of aligned high-quality samples of identities determined by features of a pretrained recognition
model. In addition, the framework incorporates a novel filter to prevent samples of privacy-breaching
identities from reaching the generated datasets and to also improve identity separability and intra-identity
diversity. Extensive experiments on six publicly available datasets reveal that our framework achieves
competitive synthesis capabilities while preserving the privacy of real-world subjects. The synthesized
datasets also facilitate training more powerful recognition models than datasets generated by competing
methods or even small-scale real-world datasets. Employing both visible and near-infrared data for training
also results in higher recognition accuracy on real-world visible spectrum benchmarks. Thus, training
with multispectral data could potentially improve existing recognition systems that utilize only the visible
spectrum, without the need for additional sensors.*

# Release Notes: 

The ArcBiFaceGAN PyTorch framework allows for the generation of large-scale recognition datasets of visible and near-infrared privacy-preserving face images. 

The framework is made up of an identity-conditioned Dual-Branch StyleGAN2, based on the [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) implementation and an auxiliary Privacy and Diversity filter, based on the pre-trained [ArcFace recognition model](https://github.com/chenggongliang/arcface).

This repository follows the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).

Additional details and instructions for running ArcBiFaceGAN will be provided in the coming days. 

# Requirements and Setup:

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have tested our implementation on a NVIDIA RTX 3060 GPU and a NVIDIA RTX 3090 GPU. Parallelization across multiple GPUs are also supported for training the DB-StyleGAN2 network.
* We highly recommend using Docker to setup the environment. Please use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies. (The Docker image requires NVIDIA driver release `r455.23` or later.)
* Otherwise the requirements remain the same as in  [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). These being 64-bit Python 3.7, PyTorch 1.7.1, and CUDA toolkit 11.0 or later. Use at least version 11.1 if running on RTX 3090. Check the linked repository if you are having any problems.


How to build the Docker environment: 
```.bash
docker build --tag sg2ada:latest .
```

# Acknowledgements

Supported in parts by the Slovenian Research and Innovation Agency ARIS through the Research Programmes P2-0250(B) "Metrology and Biometric Systems" and P2--0214 (A) “Computer Vision”, the ARIS Project J2-2501(A) "DeepBeauty" and the ARIS Young Researcher Program.

<img src="./docs/ARIS_logo_eng_resized.jpg" alt="ARIS_logo_eng_resized" width="300"/>
