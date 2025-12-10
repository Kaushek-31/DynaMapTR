DynaMapTR: Dynamic-Aware HD Map Construction
============================================

**University of Michigan | ROB 535 Final Project**

This repository contains the implementation of **DynaMapTR**, a dynamic-aware HD map construction pipeline. This method integrates **BEVFormer** and **MapTRv2** with a novel segmentation-guided BEV masking block to explicitly suppress dynamic object activations (vehicles, pedestrians) before map decoding.

Abstract
--------

High-Definition (HD) maps are crucial for autonomous driving, but current BEV-based mapping methods (like MapTR) suffer from interference caused by dynamic objects. These objects create strong feature activations that degrade transformer attention and distort static geometric predictions.

**DynaMapTR** addresses this by:

This results in more stable attention and geometrically accurate vector maps, particularly in cluttered urban scenes4.

Architecture
------------

The pipeline consists of three main stages5:

1.  **BEV Encoder:** Generates dense BEV features from multi-view images.
    
2.  **Dynamic Masking Module:** Performs 3-class segmentation (Background, Vehicle, Pedestrian) and masks the feature map6.
    
3.  **Map Decoder:** The masked features are passed to the MapTRv2 decoder to predict vector elements (lanes, dividers, crossings).
    

_(Refer to Fig 1. in the report for the architecture schematic)_ 7

Installation
------------

### Prerequisites

*   Linux
    
*   Python 3.8+
    
*   PyTorch 1.10+
    
*   CUDA 11.1+
    
*   [MapTRv2 Dependencies](https://github.com/hustvl/MapTR)
    

### Setup

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Clone the repository  git clone https://github.com/yourusername/DynaMapTR.git  cd DynaMapTR  # Install dependencies (Based on MapTR/mmdetection3d)  pip install -r requirements.txt   `

Dataset Preparation
-------------------

This project uses the **nuScenes** dataset. Please download the dataset and organize it as follows:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data/    nuscenes/      maps/      samples/      sweeps/      v1.0-trainval/   `

**Note:** For dynamic object masking, we generate custom 3-class BEV segmentation labels derived from the annotated 3D bounding boxes8.

Training Strategy
-----------------

DynaMapTR utilizes a **two-stage training strategy** to prevent gradient interference between the segmentation and mapping tasks9.

### Stage 1: Segmentation Training

In this stage, we train the BEVFormer encoder and the SegEncodeV2 head jointly. The MapTR decoder is disabled10.

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Train the segmentation head and encoder  python tools/train.py configs/dynamaptr/stage1_segmentation.py   `

*   **Objective:** Minimize FocalLoss + DiceLoss11.
    
*   **Target:** Background, Vehicles, Pedestrians.
    

### Stage 2: HD Map Training

We freeze the segmentation module to act as a stable filter. The masked BEV features are passed to the MapTR decoder12.

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Train the map decoder with frozen segmentation weights  python tools/train.py configs/dynamaptr/stage2_map_decoding.py --load_from work_dirs/stage1/latest.pth   `

Results
-------

### Segmentation Performance (Stage 1)

The lightweight segmentation module achieves the following IoU in BEV space13:

**ClassIoUBackground**0.98**Vehicles**0.65**Pedestrians**0.10

### Map Construction Quality

Qualitative results demonstrate that DynaMapTR reduces "attention drift" around vehicles and produces clearer lane boundaries compared to the baseline MapTRv214141414.

_(Comparison of Ground Truth, Plain BEV, and Masked BEV predictions - see Fig. 3 in report)_ 15

Team
----

*   **Kaushek Kumar T R** - _EECS Department_ - [kaushek@umich.edu](mailto:kaushek@umich.edu) 16161616
    
*   **Maithreyan Ganesh** - _Robotics Department_ - [maithgan@umich.edu](mailto:maithgan@umich.edu) 17171717
    
*   **Shivam Udeshi** - _EECS Department_ - [sudeshi@umich.edu](mailto:sudeshi@umich.edu) 18181818
    

**University of Michigan, Ann Arbor** 19

Acknowledgements
----------------

We thank the ROB 535 instructors for their guidance and compute resources20.

This project builds upon excellent open-source work:

*   [MapTR / MapTRv2](https://github.com/hustvl/MapTR)
    
*   [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
