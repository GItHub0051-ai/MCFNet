# MCFNet
This is a project about Multi-channel Cross-modal Fusion Network (MCFNet).
# Introduction
The MCFNet is a comprehensive framework based on cross-modal attention mechanism, which for the first time uses a multi-channel approach to apply different objects to attention.This method greatly improves the recognition rate of multimodal sentiment analysis, surpassing most of the existing best methods.
# Training
## Platform
Windows 11 OS, Nvidia GPU 2080ti.
## Settings
* Epoch: 50 for MOSI, 200 for MOSEI.
* Learning rate: 1e-5.
* dropout: 0.1
* Loss function: BCE Loss.
## Conclusion
This paper introduces the Multi-channel Cross-modal Fusion Network (MCFNet) for MSA, which facilitates the fusion process of inconsistent modalities and enhances the significance of language.
However, the following problems remain:
1)	Large data sets may be incomplete, adopting latent factor analysis can help us extract their inherent features and promote the full expression of multimodal data.
2)	Models that are too complex can be difficult to deploy industrially, so consider the computational complexity during out future design process.
3)	Deploying the proposed MCFNet model to short video applications are of great significance to help obtain real-time human sentiment and other tasks.
