# Enabling AI on the Edge
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![repo size](https://img.shields.io/github/repo-size/wangxb96/Awesome-Edge-Intelligence)](https://github.com/wangxb96/Awesome-Edge-Intelligence)
![License](https://img.shields.io/github/license/wangxb96/Awesome-Edge-Intelligence?color=blue) 

# Table of Contents
  - [1. Background Knowledge](#1-background-knowledge)
    - [1.1. Edge Computing](#11-edge-computing)
    - [1.2. Edge AI](#12-edge-ai)
      - [1.2.1. Blogs About Edge AI](#121-blogs-about-edge-ai)
  - [2. Our Survey](#2-our-survey)
  - [3. Efficient Computing Methods for Edge Intelligence](#3-efficient-computing-methods-for-edge-intelligence)
    - [3.1. Data Prepocessing](#31-data-prepocessing)
      - [3.1.1. Data Cleaning](#311-data-cleaning)
      - [3.1.2. Feature Compression](#312-feature-compression)
        - [3.1.2.1. Feature Selection](#3121-feature-selection)
        - [3.1.2.2. Feature Extraction](#3122-feature-extraction)
      - [3.1.3. Data Augmentation](#313-data-augmentation)
    - [3.2. Efficient Model Architecture](#32-efficient-model-architecture)
      - [3.2.1. Compact Architecture](#321-compact-architecture)
      - [3.2.2. Neural Architecture Search (NAS)](#322-neural-architecture-search-nas)
    - [3.3. Model Compression](#33-model-compression)
      - [3.3.1. Pruning](#331-pruning)
      - [3.3.2. Parameter Sharing](#332-parameter-sharing)
      - [3.3.3. Quantization](#333-quantization)
      - [3.3.4. Knowledge Distillation](#334-knowledge-distillation)
      - [3.3.5. Low-rank Factorization](#335-low-rank-factorization)
    - [3.4. Model Acceleration](#34-model-acceleration)
      - [3.4.1. AI Inference Framework](#341-ai-inference-framework)
      - [3.4.2. AI Model Accelerator](#342-ai-model-accelerator)
   - [4. Important Surveys on Edge AI (Related to edge inference and model deployment)](#4-important-surveys-on-edge-ai-related-to-edge-inference-and-model-deploy)
    
## 1. Background Knowledge
### 1.1. Edge Computing
Edge computing is a distributed computing paradigm that brings computation and data storage closer to the sources of data. This is expected to improve response times and save bandwidth.

- [What is edge computing? Everything you need to know](https://www.techtarget.com/searchdatacenter/definition/edge-computing)
- [Machine Learning at the Edge — μML](https://heartbeat.comet.ml/machine-learning-at-the-edge-%CE%BCml-2802f1af92de)

### 1.2. Edge AI
- Edge AI refers to the deployment of artificial intelligence (AI) algorithms and models directly on edge devices, such as mobile phones, Internet of Things (IoT) devices, and other smart sensors. 
- By processing data locally on the device rather than relying on cloud-based algorithms, edge AI enables real-time decision-making and reduces the need for data to be transmitted to remote servers. This can lead to reduced latency, improved data privacy and security, and reduced bandwidth requirements. 
- Edge AI has become increasingly important as the proliferation of edge devices continues to grow, and the need for intelligent and low-latency decision-making becomes more pressing.


#### 1.2.1. Blogs About Edge AI
- [Edge AI – What is it and how does it Work?](https://micro.ai/blog/edge-ai-what-is-it-and-how-does-it-work)
- [What is Edge AI?](https://www.advian.fi/en/what-is-edge-ai)
- [Edge AI – Driving Next-Gen AI Applications in 2022](https://viso.ai/edge-ai/edge-ai-applications-and-trends/)
- [Edge Intelligence: Edge Computing and Machine Learning (2023 Guide)](https://viso.ai/edge-ai/edge-intelligence-deep-learning-with-edge-computing/)
- [What is Edge AI, and how does it work?](https://xailient.com/blog/a-comprehensive-guide-to-edge-ai/)
- [Edge AI 101- What is it, Why is it important, and How to implement Edge AI?](https://www.seeedstudio.com/blog/2021/04/02/edge-ai-what-is-it-and-what-can-it-do-for-edge-iot/)
- [Edge AI: The Future of Artificial Intelligence](https://softtek.eu/en/tech-magazine-en/artificial-intelligence-en/edge-ai-el-futuro-de-la-inteligencia-artificial/)
- [What is Edge AI? Machine Learning + IoT](https://www.digikey.com/en/maker/projects/what-is-edge-ai-machine-learning-iot/4f655838138941138aaad62c170827af)
- [What is edge AI computing?](https://www.telusinternational.com/insights/ai-data/article/what-is-edge-ai-computing)
- [在边缘实现机器学习都需要什么？](https://www.infoq.cn/article/shdudgbwmho0ewwpmk5i)
- [边缘计算 | 在移动设备上部署深度学习模型的思路与注意点](https://www.cnblogs.com/showmeai/p/16627579.html)

## 2. Our Survey (To be released)
![Framework](https://raw.githubusercontent.com/wangxb96/Awesome-AI-on-the-Edge/main/Framework.png)



##  3. Efficient Computing Methods for Edge Intelligence
### 3.1. Data Prepocessing
#### 3.1.1. Data Cleaning
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Active label cleaning for improved dataset quality under resource constraints[J]. Nature communications, 2022.](https://www.nature.com/articles/s41467-022-28818-3) | Microsoft Research Cambridge |  [Code](https://github.com/microsoft/InnerEye-DeepLearning/tree/1606729c7a16e1bfeb269694314212b6e2737939/InnerEye-DataQuality) |
| [Locomotion mode recognition using sensory data with noisy labels: A deep learning approach. IEEE Trans. on Mobile Computing.](https://ieeexplore.ieee.org/abstract/document/9653808/) | Indian Institute of Technology BHU Varanasi | [Code](https://github.com/errahulm/LRNL_approach) |
| [Big data cleaning based on mobile edge computing in industrial sensor-cloud[J]. IEEE Trans. on Industrial Informatics, 2019.](https://ieeexplore.ieee.org/abstract/document/8822503/) | Huaqiao University | -- |
| [Federated data cleaning: Collaborative and privacy-preserving data cleaning for edge intelligence[J]. IEEE Internet of Things Journal, 2020.](https://ieeexplore.ieee.org/abstract/document/9210000/) | Xidian University | -- |
| [A data stream cleaning system using edge intelligence for smart city industrial environments[J]. IEEE Trans. on Industrial Informatics, 2021.](https://ieeexplore.ieee.org/abstract/document/9424956/) | Hangzhou Dianzi University | -- |
| [Protonn: Compressed and accurate knn for resource-scarce devices[C]//ICML, 2017.](https://proceedings.mlr.press/v70/gupta17a.html) | Microsoft Research, India | [Code](https://github.com/Microsoft/ELL) |
| [Intelligent data collaboration in heterogeneous-device iot platforms[J]. ACM Trans. on Sensor Networks (TOSN), 2021.](https://dl.acm.org/doi/abs/10.1145/3427912) | Hangzhou Dianzi University | -- |


#### 3.1.2. Feature Compression
##### 3.1.2.1. Feature Selection
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Accessible melanoma detection using smartphones and mobile image analysis[J]. IEEE Trans. on Multimedia, 2018.](https://ieeexplore.ieee.org/abstract/document/8316868/) | Singapore University of Technology and Design | -- |
| [ActID: An efficient framework for activity sensor based user identification[J]. Computers & Security, 2021.](https://sciencedirect.53yu.com/science/article/pii/S0167404821001437) | University of Houston-Clear Lake | -- |
| [Descriptor Scoring for Feature Selection in Real-Time Visual Slam[C]//ICIP, 2020.](https://cz5waila03cyo0tux1owpyofgoryroob.oss-cn-beijing.aliyuncs.com/17/B9/05/17B90503C6B685151A7C1EC364D10815.pdf) | Processor Architecture Research Lab, Intel Labs | -- |
| [Edge2Analysis: a novel AIoT platform for atrial fibrillation recognition and detection[J]. IEEE Journal of Biomedical and Health Informatics, 2022.](https://ieeexplore.ieee.org/abstract/document/9769989/) | Sun Yat-Sen University | -- |
| [Feature selection with limited bit depth mutual information for portable embedded systems[J]. Knowledge-Based Systems, 2020.](https://sci-hub.et-fine.com/10.1016/j.knosys.2020.105885) | CITIC, Universidade da Coruña | -- |
| [Seremas: Self-resilient mobile autonomous systems through predictive edge computing[C]// SECON, 2021.](https://ieeexplore.ieee.org/abstract/document/9491618/) | University of California, Irvine | -- |
| [A covid-19 detection algorithm using deep features and discrete social learning particle swarm optimization for edge computing devices[J]. ACM Trans. on Internet Technology (TOIT), 2021.](https://dl.acm.org/doi/abs/10.1145/3453170) | Hubei Province Key Laboratory of Intelligent Information Processing and Real-time Industrial System | -- |

##### 3.1.2.2. Feature Extraction
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Supervised compression for resource-constrained edge computing systems[C]// WACV, 2022.](https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html) | University of California, Irvine | [Code](https://github.com/yoshitomo-matsubara/supervised-compression) |
| ["Blessing of dimensionality: High-dimensional feature and its efficient compression for face verification." CVPR, 2013.](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Chen_Blessing_of_Dimensionality_2013_CVPR_paper.html) | University of Science and Technology of China | -- |
| [Toward intelligent sensing: Intermediate deep feature compression[J]. IEEE Trans. on Image Processing, 2019.](https://ieeexplore.ieee.org/abstract/document/8848858/) | Nangyang Technological University | -- | 
| [Selective feature compression for efficient activity recognition inference[C]// ICCV, 2021.](http://openaccess.thecvf.com/content/ICCV2021/html/Liu_Selective_Feature_Compression_for_Efficient_Activity_Recognition_Inference_ICCV_2021_paper.html) | Amazon Web Services | -- |
| [Video coding for machines: A paradigm of collaborative compression and intelligent analytics[J]. IEEE Trans. on Image Processing, 2020.](https://ieeexplore.ieee.org/abstract/document/9180095/) | Peking University | -- |
| [Communication-computation trade-off in resource-constrained edge inference[J]. IEEE Communications Magazine, 2020.](https://ieeexplore.ieee.org/abstract/document/9311935/) | The Hong Kong Polytechnic University | [Code](https://github.com/shaojiawei07/Edge_Inference_three-step_framework) |
| [Edge-based compression and classification for smart healthcare systems: Concept, implementation and evaluation[J]. Expert Systems with Applications, 2019.](https://www.sciencedirect.com/science/article/pii/S0957417418305967) | Qatar University | -- |
| [EFCam: Configuration-adaptive fog-assisted wireless cameras with reinforcement learning[C]// SECON, 2021.](https://ieeexplore.ieee.org/abstract/document/9491609/) | Nanyang Technological University | -- | 
| [Edge computing for smart health: Context-aware approaches, opportunities, and challenges[J]. IEEE Network, 2019.](https://ieeexplore.ieee.org/abstract/document/8674240/) | Qatar University | -- |
| [DEEPEYE: A deeply tensor-compressed neural network for video comprehension on terminal devices[J]. ACM Trans. on Embedded Computing Systems (TECS), 2020.](https://dl.acm.org/doi/abs/10.1145/3381805) | Shanghai Jiao Tong University | -- |
| [CROWD: crow search and deep learning based feature extractor for classification of Parkinson’s disease[J]. ACM Trans. on Internet Technology (TOIT), 2021.](https://dl.acm.org/doi/abs/10.1145/3418500) | Taif University | -- |
| ["Deep-Learning Based Monitoring Of Fog Layer Dynamics In Wastewater Pumping Stations", Water research 202 (2021): 117482.](https://cz5waila03cyo0tux1owpyofgoryroob.oss-cn-beijing.aliyuncs.com/50/65/C8/5065C803C133DF389836C73A6267F1E0.pdf) | Deltares | -- |
| [Distributed and efficient object detection via interactions among devices, edge, and cloud[J]. IEEE Trans. on Multimedia, 2019.](https://ieeexplore.ieee.org/abstract/document/8695132/) | Central South University | -- |

#### 3.1.3. Data Augmentation
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [An effective litchi detection method based on edge devices in a complex scene[J]. Biosystems Engineering, 2022.](https://sciencedirect.53yu.com/science/article/pii/S1537511022001714) | Beihang University | -- |
| [Segmentation of drivable road using deep fully convolutional residual network with pyramid pooling[J]. Cognitive Computation, 2018.](https://linkspringer.53yu.com/article/10.1007/s12559-017-9524-y) | Tsinghua University | -- |
| [Multiuser physical layer authentication in internet of things with data augmentation[J]. IEEE internet of things journal, 2019.](https://ieeexplore.ieee.org/abstract/document/8935162/) | University of Electronic Science and Technology of China | -- |
| [Data-augmentation-based cellular traffic prediction in edge-computing-enabled smart city[J]. IEEE Trans. on Industrial Informatics, 2020.](https://ieeexplore.ieee.org/abstract/document/9140397/) | University of Electronic Science and Technology of China | -- |
| [Towards light-weight and real-time line segment detection[C]// AAAI, 2022.](https://ojs.aaai.org/index.php/AAAI/article/view/19953) | NAVER/LINE Corp. | [Code](https://github.com/navervision/mlsd) |
| [Intrusion Detection System After Data Augmentation Schemes Based on the VAE and CVAE[J]. IEEE Trans. on Reliability, 2022.](https://ieeexplore.ieee.org/abstract/document/9761959/) | Guangdong Ocean University | -- |
| [Magicinput: Training-free multi-lingual finger input system using data augmentation based on mnists[C]// ICIP, 2021.](https://dl.acm.org/doi/abs/10.1145/3412382.3458261) | Shanghai Jiao Tong University | -- |


### 3.2. Efficient Model Architecture
#### 3.2.1. Compact Architecture
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.](https://arxiv.53yu.com/abs/1704.04861) | Google Inc. | [Code](https://github.com/Zehaos/MobileNet) |
| [Mobilenetv2: Inverted residuals and linear bottlenecks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 4510-4520.](http://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) | Google Inc. | [Code](https://github.com/d-li14/mobilenetv2.pytorch) |  
| [Searching for mobilenetv3[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 1314-1324.](http://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | Google Inc. | [Code](https://github.com/xiaolai-sqlai/mobilenetv3) | 
| [Rethinking bottleneck structure for efficient mobile network design[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part III 16. Springer International Publishing, 2020: 680-697.](https://arxiv.org/abs/2007.02269) | National University of Singapore | [Code](https://github.com/zhoudaquan/rethinking_bottleneck_design) |
| [Mnasnet: Platform-aware neural architecture search for mobile[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 2820-2828.](http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper) | Google Brain | [Code](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) |
| [Shufflenet: An extremely efficient convolutional neural network for mobile devices[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6848-6856.](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html) | Megvii Inc (Face++) | [Code](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1) |
| [Shufflenet v2: Practical guidelines for efficient cnn architecture design[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 116-131.](http://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html) | Megvii Inc (Face++) | [Code](https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2) |
| [Single path one-shot neural architecture search with uniform sampling[C]//Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVI 16. Springer International Publishing, 2020: 544-560.](https://linkspringer.53yu.com/chapter/10.1007/978-3-030-58517-4_32) | Megvii Inc (Face++) | [Code](https://github.com/megvii-model/ShuffleNet-Series/tree/master/OneShot) |
| [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size[J]. arXiv preprint arXiv:1602.07360, 2016.](https://arxiv.53yu.com/abs/1602.07360) | DeepScale∗ & UC Berkeley | [Code](https://github.com/forresti/SqueezeNet) |
| [Squeezenext: Hardware-aware neural network design[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2018: 1638-1647.](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w33/html/Gholami_SqueezeNext_Hardware-Aware_Neural_CVPR_2018_paper.html) | UC Berkeley | [Code](https://github.com/Timen/squeezenext-tensorflow) |
| [Ghostnet: More features from cheap operations[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 1580-1589.](http://openaccess.thecvf.com/content_CVPR_2020/html/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.html) | Noah’s Ark Lab, Huawei Technologies | [Code](https://github.com/huawei-noah/ghostnet) |
| [Efficientnet: Rethinking model scaling for convolutional neural networks[C]//International conference on machine learning. PMLR, 2019: 6105-6114.](http://proceedings.mlr.press/v97/tan19a.html) | Google Brain | [Code](https://github.com/lukemelas/EfficientNet-PyTorch) |
| [Efficientnetv2: Smaller models and faster training[C]//International conference on machine learning. PMLR, 2021: 10096-10106.](http://proceedings.mlr.press/v139/tan21a.html) | Google Brain | [Code](https://github.com/google/automl/tree/master/efficientnetv2) |
| [Efficientdet: Scalable and efficient object detection[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 10781-10790.](http://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html) | Google Brain | [Code](https://github.com/google/automl/tree/master/efficientdet) |
| [Condensenet: An efficient densenet using learned group convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 2752-2761.](http://openaccess.thecvf.com/content_cvpr_2018/html/Huang_CondenseNet_An_Efficient_CVPR_2018_paper.html) | Cornell University | [Code](https://github.com/ShichenLiu/CondenseNet) |
| [Condensenet v2: Sparse feature reactivation for deep networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 3569-3578.](http://openaccess.thecvf.com/content/CVPR2021/html/Yang_CondenseNet_V2_Sparse_Feature_Reactivation_for_Deep_Networks_CVPR_2021_paper.html) | Tsinghua University |[Code](https://github.com/jianghaojun/CondenseNetV2) |
| [Espnet: Efficient spatial pyramid of dilated convolutions for semantic segmentation[C]//Proceedings of the european conference on computer vision (ECCV). 2018: 552-568.](http://openaccess.thecvf.com/content_ECCV_2018/html/Sachin_Mehta_ESPNet_Efficient_Spatial_ECCV_2018_paper.html) | University of Washington | [Code](https://github.com/sacmehta/ESPNet/) |
| [Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019: 9190-9200.](http://openaccess.thecvf.com/content_CVPR_2019/html/Mehta_ESPNetv2_A_Light-Weight_Power_Efficient_and_General_Purpose_Convolutional_Neural_CVPR_2019_paper.html) | University of Washington | [Code](https://github.com/sacmehta/ESPNetv2) |
| [Fbnet: Hardware-aware efficient convnet design via differentiable neural architecture search[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 10734-10742.](http://openaccess.thecvf.com/content_CVPR_2019/html/Wu_FBNet_Hardware-Aware_Efficient_ConvNet_Design_via_Differentiable_Neural_Architecture_Search_CVPR_2019_paper.html) | UC Berkeley | [Code](https://github.com/facebookresearch/mobile-vision) |
| [Fbnetv2: Differentiable neural architecture search for spatial and channel dimensions[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 12965-12974.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wan_FBNetV2_Differentiable_Neural_Architecture_Search_for_Spatial_and_Channel_Dimensions_CVPR_2020_paper.pdf) | UC Berkeley | [Code](https://github.com/facebookresearch/mobile-vision) |
| [Fbnetv3: Joint architecture-recipe search using predictor pretraining[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 16276-16285.](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_FBNetV3_Joint_Architecture-Recipe_Search_Using_Predictor_Pretraining_CVPR_2021_paper.pdf) | Facebook Inc. & UC Berkeley | -- |
| [Pelee: A real-time object detection system on mobile devices[J]. Advances in neural information processing systems, 2018, 31.](https://proceedings.neurips.cc/paper/2018/file/9908279ebbf1f9b250ba689db6a0222b-Paper.pdf) | University of Western Ontario | [Code](https://github.com/Robert-JunWang/Pelee) |
| [Going deeper with convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 1-9.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) | Google Inc. | [Code](https://github.com/conan7882/GoogLeNet-Inception) |
| [Batch normalization: Accelerating deep network training by reducing internal covariate shift[C]//International conference on machine learning. pmlr, 2015: 448-456.](https://arxiv.org/pdf/1502.03167.pdf) | Google Inc. | [Code](https://github.com/shanglianlm0525/PyTorch-Networks/blob/master/ClassicNetwork/InceptionV2.py) |
| [Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 2818-2826.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf) | Google Inc. | [Code](https://github.com/shanglianlm0525/PyTorch-Networks/blob/master/ClassicNetwork/InceptionV3.py) |
| [Inception-v4, inception-resnet and the impact of residual connections on learning[C]//Proceedings of the AAAI conference on artificial intelligence. 2017, 31(1).](https://ojs.aaai.org/index.php/aaai/article/view/11231) | Google Inc. | [Code](https://github.com/shanglianlm0525/PyTorch-Networks/blob/master/ClassicNetwork/InceptionV4.py) |
| [Xception: Deep learning with depthwise separable convolutions[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 1251-1258.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf) | Google, Inc. | [Code](https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/xception.py) |
| [Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer[J]. arXiv preprint arXiv:2110.02178, 2021.](https://arxiv.org/pdf/2110.02178.pdf) | Apple | [Code](https://github.com/apple/ml-cvnets) |
| [Lite transformer with long-short range attention[J]. arXiv preprint arXiv:2004.11886, 2020.](https://arxiv.org/pdf/2004.11886.pdf) | Massachusetts Institute of Technology | [Code](https://github.com/mit-han-lab/lite-transformer) |
| [Coordinate attention for efficient mobile network design[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 13713-13722.](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.pdf) | National University of Singapore | [Code](https://github.com/houqb/CoordAttention) |
| [ECA-Net: Efficient channel attention for deep convolutional neural networks[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 11534-11542.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf) | Tianjin University | [Code](https://github.com/BangguWu/ECANet) |
| [Sa-net: Shuffle attention for deep convolutional neural networks[C]//ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021: 2235-2239.](https://ieeexplore.ieee.org/abstract/document/9414568) | Nanjing University | [Code](https://github.com/wofmanaf/SA-Net) |
| [Triplet Attention: Rethinking the Similarity in Transformers[C]//Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021: 2378-2388.](https://dl.acm.org/doi/abs/10.1145/3447548.3467241) | Beihang University | [Code](https://github.com/zhouhaoyi/TripletAttention) |
| [Resnest: Split-attention networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 2736-2746.](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf) | Meta | [Code](https://github.com/zhanghang1989/ResNeSt) |    

#### 3.2.2. Neural Architecture Search (NAS)
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [FTT-NAS: Discovering fault-tolerant convolutional neural architecture[J]. ACM Transactions on Design Automation of Electronic Systems (TODAES), 2021, 26(6): 1-24.](https://dl.acm.org/doi/abs/10.1145/3460288) | Tsinghua University | [Code](https://github.com/walkerning/aw_nas) |
| [An adaptive neural architecture search design for collaborative edge-cloud computing[J]. IEEE Network, 2021, 35(5): 83-89.](https://ieeexplore.ieee.org/abstract/document/9606812/) | Nanjing University of Posts and Telecommunications | -- |
| [Binarized neural architecture search for efficient object recognition[J]. International Journal of Computer Vision, 2021, 129: 501-516.](https://linkspringer.53yu.com/article/10.1007/s11263-020-01379-y) | Beihang University | -- |
| [Multiobjective reinforcement learning-based neural architecture search for efficient portrait parsing[J]. IEEE Transactions on Cybernetics, 2021.](https://ieeexplore.ieee.org/abstract/document/9524839/) | University of Electronic Science and Technology of China | -- |
| [Intermittent-aware neural architecture search[J]. ACM Transactions on Embedded Computing Systems (TECS), 2021, 20(5s): 1-27.](https://dl.acm.org/doi/abs/10.1145/3476995) | Academia Sinica and National Taiwan University | [Code](https://github.com/EMCLab-Sinica/Intermittent-aware-NAS) |
| [Hardcore-nas: Hard constrained differentiable neural architecture search[C]//International Conference on Machine Learning. PMLR, 2021: 7979-7990.](http://proceedings.mlr.press/v139/nayman21a.html) | Alibaba Group, Tel Aviv, Israel | [Code](https://github.com/Alibaba-MIIL/HardCoReNAS) |
| [MemNAS: Memory-efficient neural architecture search with grow-trim learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 2108-2116.](http://openaccess.thecvf.com/content_CVPR_2020/html/Liu_MemNAS_Memory-Efficient_Neural_Architecture_Search_With_Grow-Trim_Learning_CVPR_2020_paper.html) | Beijing University of Posts and Telecommunications | -- |
| [Pvnas: 3D neural architecture search with point-voxel convolution[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(11): 8552-8568.](https://ieeexplore.ieee.org/abstract/document/9527118/) | Massachusetts Institute of Technology | -- |
| [Toward tailored models on private aiot devices: Federated direct neural architecture search[J]. IEEE Internet of Things Journal, 2022, 9(18): 17309-17322.](https://ieeexplore.ieee.org/abstract/document/9721425/) | Northeastern University, Qinhuangdao | -- |
| [Automatic design of convolutional neural network architectures under resource constraints[J]. IEEE Transactions on Neural Networks and Learning Systems, 2021.](https://ieeexplore.ieee.org/abstract/document/9609007/) | Sichuan University | -- |


### 3.3. Model Compression
#### 3.3.1. Pruning
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Supervised compression for resource-constrained edge computing systems[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2022: 2685-2695.](https://openaccess.thecvf.com/content/WACV2022/html/Matsubara_Supervised_Compression_for_Resource-Constrained_Edge_Computing_Systems_WACV_2022_paper.html) | University of Pittsburgh | -- |
| [Train big, then compress: Rethinking model size for efficient training and inference of transformers[C]//International Conference on machine learning. PMLR, 2020: 5958-5968.](http://proceedings.mlr.press/v119/li20m.html) | UC Berkeley | -- |
| [Hrank: Filter pruning using high-rank feature map[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 1529-1538.](http://openaccess.thecvf.com/content_CVPR_2020/html/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.html) |  Xiamen University | [Code](https://github.com/lmbxmu/HRank) | 
| [Clip-q: Deep network compression learning by in-parallel pruning-quantization[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7873-7882.](http://openaccess.thecvf.com/content_cvpr_2018/html/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.html) | Simon Fraser University | -- |
| Sparse: Sparse architecture search for cnns on resource-constrained microcontrollers[J]. Advances in Neural Information Processing Systems, 2019, 32.](https://proceedings.neurips.cc/paper/2019/hash/044a23cadb567653eb51d4eb40acaa88-Abstract.html) | Arm ML Research | -- |
| [Deepadapter: A collaborative deep learning framework for the mobile web using context-aware network pruning[C]//IEEE INFOCOM 2020-IEEE Conference on Computer Communications. IEEE, 2020: 834-843.](https://ieeexplore.ieee.org/abstract/document/9155379/) |  Beijing University of Posts and Telecommunications | -- |
| [SCANN: Synthesis of compact and accurate neural networks[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2021, 41(9): 3012-3025.](https://ieeexplore.ieee.org/abstract/document/9552199/) | Princeton University | -- |
| [Directx: Dynamic resource-aware cnn reconfiguration framework for real-time mobile applications[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020, 40(2): 246-259.](https://ieeexplore.ieee.org/abstract/document/9097286/) | George Mason University | -- |
| [Pruning deep reinforcement learning for dual user experience and storage lifetime improvement on mobile devices[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020, 39(11): 3993-4005.](https://ieeexplore.ieee.org/abstract/document/9211447/) | City University of Hong Kong | -- |
| [SuperSlash: A unified design space exploration and model compression methodology for design of deep learning accelerators with reduced off-chip memory access volume[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020, 39(11): 4191-4204.](https://ieeexplore.ieee.org/abstract/document/9211496/) | Information Technology University | -- |
| [Penni: Pruned kernel sharing for efficient CNN inference[C]//International Conference on Machine Learning. PMLR, 2020: 5863-5873.](http://proceedings.mlr.press/v119/li20d.html) | Duke University | [Code](https://github.com/timlee0212/PENNI) | 
| [Fast operation mode selection for highly efficient iot edge devices[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2019, 39(3): 572-584.](https://ieeexplore.ieee.org/abstract/document/8634947) | Karlsruhe Institute of Technology | -- |
| [Efficient on-chip learning for optical neural networks through power-aware sparse zeroth-order optimization[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(9): 7583-7591.](https://ojs.aaai.org/index.php/AAAI/article/view/16928) | University of Texas at Austin | -- |
| [A Fast Post-Training Pruning Framework for Transformers[C]//Advances in Neural Information Processing Systems.](https://arxiv.org/pdf/2204.09656.pdf) | UC Berkeley |  [Code](https://github.com/WoosukKwon/retraining-free-pruning) |
| [Radio frequency fingerprinting on the edge[J]. IEEE Transactions on Mobile Computing, 2021, 21(11): 4078-4093.](https://ieeexplore.ieee.org/abstract/document/9372779/) | Northeastern University, Boston | -- |
| [Exploring sparsity in image super-resolution for efficient inference[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 4917-4926.](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Exploring_Sparsity_in_Image_Super-Resolution_for_Efficient_Inference_CVPR_2021_paper.html?ref=https://githubhelp.com) | National University of Defense Technology | [Code](https://github.com/LongguangWang/SMSR) |
| [O3BNN-R: An out-of-order architecture for high-performance and regularized BNN inference[J]. IEEE Transactions on parallel and distributed systems, 2020, 32(1): 199-213.](https://ieeexplore.ieee.org/abstract/document/9154597/) | Boston University | -- |
| [Enabling on-device cnn training by self-supervised instance filtering and error map pruning[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020, 39(11): 3445-3457.](https://ieeexplore.ieee.org/abstract/document/9211513/) | University of Pittsburgh | -- |
| [Dropnet: Reducing neural network complexity via iterative pruning[C]//International Conference on Machine Learning. PMLR, 2020: 9356-9366.](http://proceedings.mlr.press/v119/tan20a.html) | National University of Singapore | [Code](https://github.com/tanchongmin/DropNet) |
| [Edgebert: Sentence-level energy optimizations for latency-aware multi-task nlp inference[C]//MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture. 2021: 830-844.](https://dl.acm.org/doi/abs/10.1145/3466752.3480095) | Harvard University | [Code](https://github.com/harvard-acc/EdgeBERT) |
| [Fusion-catalyzed pruning for optimizing deep learning on intelligent edge devices[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020, 39(11): 3614-3626.](https://ieeexplore.ieee.org/abstract/document/9211462/) | Chinese Academy of Sciences | -- |
| [3D CNN acceleration on FPGA using hardware-aware pruning[C]//2020 57th ACM/IEEE Design Automation Conference (DAC). IEEE, 2020: 1-6.](https://ieeexplore.ieee.org/abstract/document/9218571/) | Northeastern University, MA | -- |
| [Width & depth pruning for vision transformers[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(3): 3143-3151.](https://ojs.aaai.org/index.php/AAAI/article/view/20222) | Institute of Computing Technology, Chinese Academy of Sciences | -- |
| [Prive-hd: Privacy-preserved hyperdimensional computing[C]//2020 57th ACM/IEEE Design Automation Conference (DAC). IEEE, 2020: 1-6.](https://ieeexplore.ieee.org/abstract/document/9218493/) | UC San Diego | -- |
| [http://openaccess.thecvf.com/content/CVPR2021/html/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.html](http://openaccess.thecvf.com/content/CVPR2021/html/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.html) | Princeton University | -- |
| [NestFL: efficient federated learning through progressive model pruning in heterogeneous edge computing[C]//Proceedings of the 28th Annual International Conference on Mobile Computing And Networking. 2022: 817-819.](https://dl.acm.org/doi/abs/10.1145/3495243.3558248) | Purple Mountain Laboratories, Nanjing | -- |


### 3.3.2. Parameter Sharing 
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Deep k-means: Re-training and parameter sharing with harder cluster assignments for compressing deep convolutions[C]//International Conference on Machine Learning. PMLR, 2018: 5363-5372.](http://proceedings.mlr.press/v80/wu18h.html) | Texas A&M University | [Code](https://github.com/Sandbox3aster/Deep-K-Means) |
| [T-basis: a compact representation for neural networks[C]//International Conference on Machine Learning. PMLR, 2020: 7392-7404.](http://proceedings.mlr.press/v119/obukhov20a/obukhov20a.pdf) | ETH Zurich | [Code](http://obukhov.ai/tbasis) |
| [ "Soft Weight-Sharing for Neural Network Compression." International Conference on Learning Representations.](https://arxiv.org/pdf/1702.04008.pdf) | University of Amsterdam | [Code](https://github.com/KarenUllrich/Tutorial-SoftWeightSharingForNNCompression) |
| [ShiftAddNAS: Hardware-inspired search for more accurate and efficient neural networks[C]//International Conference on Machine Learning. PMLR, 2022: 25566-25580.](https://proceedings.mlr.press/v162/you22a.html) | Rice University | [Code](https://github.com/RICE-EIC/ShiftAddNAS) |
| [EfficientTDNN: Efficient architecture search for speaker recognition[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2022, 30: 2267-2279.](https://ieeexplore.ieee.org/abstract/document/9798861/) | Tongji University | [Code](https://github.com/mechanicalsea/sugar) |
| [A generic network compression framework for sequential recommender systems[C]//Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2020: 1299-1308.](https://dl.acm.org/doi/abs/10.1145/3397271.3401125) | University of Science and Technology | [Code](https://github.com/siat-nlp/CpRec) |
| [Neural architecture search for LF-MMI trained time delay neural networks[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2022, 30: 1093-1107.](https://ieeexplore.ieee.org/abstract/document/9721103/) | The Chinese University of Hong Kong | -- |
| [Structured transforms for small-footprint deep learning[J]. Advances in Neural Information Processing Systems, 2015, 28.](https://proceedings.neurips.cc/paper/2015/hash/851300ee84c2b80ed40f51ed26d866fc-Abstract.html) | Google, New York | -- | 


### 3.3.3. Quantization
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Fractrain: Fractionally squeezing bit savings both temporally and spatially for efficient dnn training[J]. Advances in Neural Information Processing Systems, 2020, 33: 12127-12139.](https://proceedings.neurips.cc/paper_files/paper/2020/file/8dc5983b8c4ef1d8fcd5f325f9a65511-Paper.pdf) | Rice University | [Code](https://github.com/RICE-EIC/FracTrain) |
| [Edgebert: Sentence-level energy optimizations for latency-aware multi-task nlp inference[C]//MICRO-54: 54th Annual IEEE/ACM International Symposium on Microarchitecture. 2021: 830-844.](https://dl.acm.org/doi/abs/10.1145/3466752.3480095) | Harvard University | -- |
| [Stochastic precision ensemble: self-knowledge distillation for quantized deep neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(8): 6794-6802.](https://ojs.aaai.org/index.php/AAAI/article/view/16839) | Seoul National University | -- |
| [Q-capsnets: A specialized framework for quantizing capsule networks[C]//2020 57th ACM/IEEE Design Automation Conference (DAC). IEEE, 2020: 1-6.](https://ieeexplore.ieee.org/abstract/document/9218746/) | Technische Universitat Wien (TU Wien) | [Code](https://git.io/JvDIF) |
| [Fspinn: An optimization framework for memory-efficient and energy-efficient spiking neural networks[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2020, 39(11): 3601-3613.](https://ieeexplore.ieee.org/abstract/document/9211568/) | Technische Universität Wien | -- |
| [Octo: INT8 Training with Loss-aware Compensation and Backward Quantization for Tiny On-device Learning[C]//USENIX Annual Technical Conference. 2021: 177-191.](https://www.usenix.org/system/files/atc21-zhou.pdf) | Hong Kong Polytechnic University | [Code](https://github.com/kimihe/Octo) |
| [Hardware-centric automl for mixed-precision quantization[J]. International Journal of Computer Vision, 2020, 128(8-9): 2035-2048.](https://linkspringer.53yu.com/article/10.1007/s11263-020-01339-6) | Massachusetts Institute of Technology | -- |
| [An automated quantization framework for high-utilization rram-based pim[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2021, 41(3): 583-596.](https://ieeexplore.ieee.org/abstract/document/9360862/) | Capital Normal University | -- |
| [Exact neural networks from inexact multipliers via fibonacci weight encoding[C]//2021 58th ACM/IEEE Design Automation Conference (DAC). IEEE, 2021: 805-810.](https://ieeexplore.ieee.org/abstract/document/9586245/) | Swiss Federal Institute of Technology Lausanne (EPFL) | -- |
| [Integer-arithmetic-only certified robustness for quantized neural networks[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 7828-7837.](http://openaccess.thecvf.com/content/ICCV2021/html/Lin_Integer-Arithmetic-Only_Certified_Robustness_for_Quantized_Neural_Networks_ICCV_2021_paper.html) | University of Southern California | -- |
| [Bits-Ensemble: Toward Light-Weight Robust Deep Ensemble by Bits-Sharing[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2022, 41(11): 4397-4408.](https://ieeexplore.ieee.org/abstract/document/9854091/) | McGill University | -- |
| [Similarity-Aware CNN for Efficient Video Recognition at the Edge[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2021, 41(11): 4901-4914.](https://ieeexplore.ieee.org/abstract/document/9656540/) | University of Southampton | -- |
| [ Data-Free Network Compression via Parametric Non-uniform Mixed Precision Quantization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 450-459.](http://openaccess.thecvf.com/content/CVPR2022/html/Chikin_Data-Free_Network_Compression_via_Parametric_Non-Uniform_Mixed_Precision_Quantization_CVPR_2022_paper.html) | Huawei Noah's Ark Lab | -- |

### 3.3.4. Knowledge Distillation
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Be your own teacher: Improve the performance of convolutional neural networks via self distillation[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 3713-3722.](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.pdf) | Tsinghua University | [Code](https://github.com/ArchipLab-LinfengZhang/) |
| [Dynabert: Dynamic bert with adaptive width and depth[J]. Advances in Neural Information Processing Systems, 2020, 33: 9782-9793.](https://proceedings.neurips.cc/paper/2020/hash/6f5216f8d89b086c18298e043bfe48ed-Abstract.html) | Huawei Noah’s Ark Lab | [Code](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT) |
| [Scan: A scalable neural networks framework towards compact and efficient models[J]. Advances in Neural Information Processing Systems, 2019, 32.](https://proceedings.neurips.cc/paper/2019/hash/934b535800b1cba8f96a5d72f72f1611-Abstract.html) | Tsinghua University | [Code](https://github.com/ArchipLab-LinfengZhang/pytorch-scalable-neural-networks) |
| [Content-aware gan compression[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 12156-12166.](http://openaccess.thecvf.com/content/CVPR2021/html/Liu_Content-Aware_GAN_Compression_CVPR_2021_paper.html) | Princeton University | -- |
| [Stochastic precision ensemble: self-knowledge distillation for quantized deep neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(8): 6794-6802.](https://ojs.aaai.org/index.php/AAAI/article/view/16839) | Seoul National University | -- |
| [Cross-modal knowledge distillation for vision-to-sensor action recognition[C]//ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 4448-4452.](https://ieeexplore.ieee.org/abstract/document/9746752/) | Texas State University | -- |
| [Learning efficient and accurate detectors with dynamic knowledge distillation in remote sensing imagery[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021, 60: 1-19.](https://ieeexplore.ieee.org/abstract/document/9625952/) | Chinese Academy of Sciences | -- |
| [On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation[C]//Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022: 546-555.](https://dl.acm.org/doi/abs/10.1145/3477495.3531775) | The University of Queensland | [Code](https://github.com/xiaxin1998/OD-Rec) |
| [Personalized edge intelligence via federated self-knowledge distillation[J]. IEEE Transactions on Parallel and Distributed Systems, 2022, 34(2): 567-580.](https://ieeexplore.ieee.org/abstract/document/9964434/) | Huazhong University of Science and Technology | -- |
| [Mobilefaceswap: A lightweight framework for video face swapping[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2022, 36(3): 2973-2981.](https://ojs.aaai.org/index.php/AAAI/article/view/20203) | Baidu Inc. | -- |
| [Dynamically pruning segformer for efficient semantic segmentation[C]//ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022: 3298-3302.](https://ieeexplore.ieee.org/abstract/document/9747634) | Amazon Halo Health & Wellness | -- |
| [CDFKD-MFS: Collaborative Data-Free Knowledge Distillation via Multi-Level Feature Sharing[J]. IEEE Transactions on Multimedia, 2022, 24: 4262-4274.](https://ieeexplore.ieee.org/abstract/document/9834142/) | Beijing Institute of Technology | [Code](https://github.com/Hao840/CDFKD-MFS) |
| [Learning Efficient Vision Transformers via Fine-Grained Manifold Distillation[J]. Advances in Neural Information Processing Systems, 2022, 35: 9164-9175.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3bd2d73b4e96b0ac5a319be58a96016c-Abstract-Conference.html) | Beijing Institute of Technology | [Code](https://github.com/Hao840/manifold-distillation) |
| [Learning Accurate, Speedy, Lightweight CNNs via Instance-Specific Multi-Teacher Knowledge Distillation for Distracted Driver Posture Identification[J]. IEEE Transactions on Intelligent Transportation Systems, 2022, 23(10): 17922-17935.](https://ieeexplore.ieee.org/abstract/document/9750058/) | Hefei Institutes of Physical Science (HFIPS), Chinese Academy of Sciences | -- | 

### 3.3.5. Low-rank Factorization
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Learning low-rank deep neural networks via singular vector orthogonality regularization and singular value sparsification[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops. 2020: 678-679.](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Yang_Learning_Low-Rank_Deep_Neural_Networks_via_Singular_Vector_Orthogonality_Regularization_CVPRW_2020_paper.pdf) | Duke University | -- |
| [MicroNet: Towards image recognition with extremely low FLOPs[J]. arXiv preprint arXiv:2011.12289, 2020.](https://arxiv.org/pdf/2011.12289.pdf) | UC San Diego | -- |
| [Locality Sensitive Hash Aggregated Nonlinear Neighborhood Matrix Factorization for Online Sparse Big Data Analysis[J]. ACM/IMS Transactions on Data Science (TDS), 2022, 2(4): 1-27.](https://dl.acm.org/doi/abs/10.1145/3497749) | Hunan University | -- |




### 3.4. Model Acceleration
#### 3.4.1. AI Inference Framework
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Hidet: Task-mapping programming paradigm for deep learning tensor programs[C]//Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 2023: 370-384.](https://dl.acm.org/doi/abs/10.1145/3575693.3575702) | University of Toronto | [Code](https://www.github.com/hidet-org/hidet) |
| [SparkNoC: An energy-efficiency FPGA-based accelerator using optimized lightweight CNN for edge computing[J]. Journal of Systems Architecture, 2021, 115: 101991.](https://sciencedirect.53yu.com/science/article/pii/S1383762121000138) | Shanghai Advanced Research Institute, Chinese Academy of Sciences | -- |
| [Re-architecting the on-chip memory sub-system of machine-learning accelerator for embedded devices[C]//2016 IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2016: 1-6.](https://ieeexplore.ieee.org/abstract/document/7827590/) | Institute of Computing Technology, Chinese Academy of Sciences | -- |
| [A unified optimization approach for cnn model inference on integrated gpus[C]//Proceedings of the 48th International Conference on Parallel Processing. 2019: 1-10.](https://dl.acm.org/doi/pdf/10.1145/3337821.3337839) | Amazon Web Services | [Code](https://github.com/dmlc/tvm/) |
| [ACG-engine: An inference accelerator for content generative neural networks[C]//2019 IEEE/ACM International Conference on Computer-Aided Design (ICCAD). IEEE, 2019: 1-7.](https://ieeexplore.ieee.org/abstract/document/8942169/) | University of Chinese Academy of Sciences | -- |
| [Edgeeye: An edge service framework for real-time intelligent video analytics[C]//Proceedings of the 1st international workshop on edge systems, analytics and networking. 2018: 1-6.](https://dl.acm.org/doi/pdf/10.1145/3213344.3213345) | University of Wisconsin-Madison | -- |
| [Haq: Hardware-aware automated quantization with mixed precision[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 8612-8620.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) | Massachusetts Institute of Technology | -- |
| [Source compression with bounded dnn perception loss for iot edge computer vision[C]//The 25th Annual International Conference on Mobile Computing and Networking. 2019: 1-16.](https://dl.acm.org/doi/abs/10.1145/3300061.3345448) | Hewlett Packard Labs | -- |
| [A lightweight collaborative deep neural network for the mobile web in edge cloud[J]. IEEE Transactions on Mobile Computing, 2020, 21(7): 2289-2305.](https://ieeexplore.ieee.org/abstract/document/9286558/) | Beijing University of Posts and Telecommunications | -- |
| [Enabling incremental knowledge transfer for object detection at the edge[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020: 396-397.](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Farhadi_Enabling_Incremental_Knowledge_Transfer_for_Object_Detection_at_the_Edge_CVPRW_2020_paper.pdf) | Arizona State university | -- |
| [DA3: Dynamic Additive Attention Adaption for Memory-Efficient On-Device Multi-Domain Learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 2619-2627.](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Yang_DA3_Dynamic_Additive_Attention_Adaption_for_Memory-Efficient_On-Device_Multi-Domain_Learning_CVPRW_2022_paper.pdf) | Arizona State University | -- |
| [An efficient GPU-accelerated inference engine for binary neural network on mobile phones[J]. Journal of Systems Architecture, 2021, 117: 102156.](https://sciencedirect.53yu.com/science/article/pii/S1383762121001120) | Sun Yat-sen University | [Code](https://code.ihub.org.cn/projects/915/repository/PhoneBit) |
| [RAPID-RL: A Reconfigurable Architecture with Preemptive-Exits for Efficient Deep-Reinforcement Learning[C]//2022 International Conference on Robotics and Automation (ICRA). IEEE, 2022: 7492-7498.](https://ieeexplore.ieee.org/abstract/document/9812320) | Purdue University | -- |
| [A variational information bottleneck based method to compress sequential networks for human action recognition[C]//Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2021: 2745-2754.](https://openaccess.thecvf.com/content/WACV2021/papers/Srivastava_A_Variational_Information_Bottleneck_Based_Method_to_Compress_Sequential_Networks_WACV_2021_paper.pdf) | Indian Institute of Technology Delhi | -- |
| [EdgeDRNN: Recurrent neural network accelerator for edge inference[J]. IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 2020, 10(4): 419-432.](https://ieeexplore.ieee.org/abstract/document/9268992/) | University of Zürich and ETH Zürich | -- |
| [Structured pruning of recurrent neural networks through neuron selection[J]. Neural Networks, 2020, 123: 134-141.](https://www.sciencedirect.com/science/article/pii/S0893608019303776) | University of Electronic Science and Technology of China | -- |
| [Dynamically hierarchy revolution: dirnet for compressing recurrent neural network on mobile devices[J]. arXiv preprint arXiv:1806.01248, 2018.](https://arxiv.org/pdf/1806.01248.pdf) | Arizona State University | -- | 
| [High-throughput cnn inference on embedded arm big. little multicore processors[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2019, 39(10): 2254-2267.](https://ieeexplore.ieee.org/abstract/document/8852739/) | National University of Singapore | -- |
| [SCA: a secure CNN accelerator for both training and inference[C]//2020 57th ACM/IEEE Design Automation Conference (DAC). IEEE, 2020: 1-6.](https://ieeexplore.ieee.org/abstract/document/9218752/) | University of Pittsburgh | -- |
| [NeuLens: spatial-based dynamic acceleration of convolutional neural networks on edge[C]//Proceedings of the 28th Annual International Conference on Mobile Computing And Networking. 2022: 186-199.](https://dl.acm.org/doi/abs/10.1145/3495243.3560528?casa_token=ioLt8xczh7cAAAAA:EpjRGtBuz0Hy3sYw9H4v1TVXcX03I68KMtlLjk1Tt2FhheVS0MA97woEWGgg_pjfjXc_njTf2JV8sQ) | New Jersey Institute of Technology | -- |
| [Weightless neural networks for efficient edge inference[C]//Proceedings of the International Conference on Parallel Architectures and Compilation Techniques. 2022: 279-290.](https://dl.acm.org/doi/pdf/10.1145/3559009.3569680) | The University of Texas at Austin | [Code](https://github.com/ZSusskind/BTHOWeN) |
| [O3BNN-R: An out-of-order architecture for high-performance and regularized BNN inference[J]. IEEE Transactions on parallel and distributed systems, 2020, 32(1): 199-213.](https://ieeexplore.ieee.org/abstract/document/9154597/) | -- |
| [Blockgnn: Towards efficient gnn acceleration using block-circulant weight matrices[C]//2021 58th ACM/IEEE Design Automation Conference (DAC). IEEE, 2021: 1009-1014.](https://ieeexplore.ieee.org/abstract/document/9586181/) | Peking University | -- |
| [{Hardware/Software}{Co-Programmable} Framework for Computational {SSDs} to Accelerate Deep Learning Service on {Large-Scale} Graphs[C]//20th USENIX Conference on File and Storage Technologies (FAST 22). 2022: 147-164.](https://www.usenix.org/conference/fast22/presentation/kwon) | KAIST | -- |
| [Achieving full parallelism in LSTM via a unified accelerator design[C]//2020 IEEE 38th International Conference on Computer Design (ICCD). IEEE, 2020: 469-477.](https://ieeexplore.ieee.org/abstract/document/9283568/) | University of Pittsburgh | -- |
| [Pasgcn: An reram-based pim design for gcn with adaptively sparsified graphs[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2022, 42(1): 150-163.](https://ieeexplore.ieee.org/abstract/document/9774869/) | Shanghai Jiao Tong University | -- |


#### 3.4.2. AI Model Accelerator
| Title & Basic Information | Affiliation | Code |
| ---- | ---- | ---- | 
| [Ncpu: An embedded neural cpu architecture on resource-constrained low power devices for real-time end-to-end performance[C]//2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2020: 1097-1109.](https://ieeexplore.ieee.org/abstract/document/9251958/) | Northwestern Univeristy, Evanston, IL | -- |
| [Reduct: Keep it close, keep it cool!: Efficient scaling of dnn inference on multi-core cpus with near-cache compute[C]//2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA). IEEE, 2021: 167-180.](https://ieeexplore.ieee.org/abstract/document/9499927/) | ETH Zurich  | -- |
| [FARNN: FPGA-GPU hybrid acceleration platform for recurrent neural networks[J]. IEEE Transactions on Parallel and Distributed Systems, 2021, 33(7): 1725-1738.](https://ieeexplore.ieee.org/abstract/document/9600618/) | Sungkyunkwan University | -- |
| [Apgan: Approximate gan for robust low energy learning from imprecise components[J]. IEEE Transactions on Computers, 2019, 69(3): 349-360.](https://ieeexplore.ieee.org/abstract/document/8880521/) | University of Central Florida | -- |
| [An FPGA overlay for CNN inference with fine-grained flexible parallelism[J]. ACM Transactions on Architecture and Code Optimization (TACO), 2022, 19(3): 1-26.](https://dl.acm.org/doi/full/10.1145/3519598) | International Institute of Information Technology | -- |
| [Pipelined data-parallel CPU/GPU scheduling for multi-DNN real-time inference[C]//2019 IEEE Real-Time Systems Symposium (RTSS). IEEE, 2019: 392-405.](https://ieeexplore.ieee.org/abstract/document/9052147/) | University of California, Riverside | -- |
| [Deadline-based scheduling for GPU with preemption support[C]//2018 IEEE Real-Time Systems Symposium (RTSS). IEEE, 2018: 119-130.](https://ieeexplore.ieee.org/abstract/document/8603197) | University of Modena and Reggio Emilia Modena | -- |
| [Energon: Toward Efficient Acceleration of Transformers Using Dynamic Sparse Attention[J]. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2022, 42(1): 136-149.](https://ieeexplore.ieee.org/abstract/document/9763839/) | Peking University | -- |
| [Light-OPU: An FPGA-based overlay processor for lightweight convolutional neural networks[C]//Proceedings of the 2020 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays. 2020: 122-132.](https://dl.acm.org/doi/abs/10.1145/3373087.3375311) | University of california, Los Angeles | -- |
| [Fluid Batching: Exit-Aware Preemptive Serving of Early-Exit Neural Networks on Edge NPUs[J]. arXiv preprint arXiv:2209.13443, 2022.](https://arxiv.org/pdf/2209.13443.pdf) | Samsung AI Center, Cambridge | -- |
| [BitSystolic: A 26.7 TOPS/W 2b~ 8b NPU with configurable data flows for edge devices[J]. IEEE Transactions on Circuits and Systems I: Regular Papers, 2020, 68(3): 1134-1145.](https://ieeexplore.ieee.org/abstract/document/9301197/) | Duke University | -- |
| [PL-NPU: An Energy-Efficient Edge-Device DNN Training Processor With Posit-Based Logarithm-Domain Computing[J]. IEEE Transactions on Circuits and Systems I: Regular Papers, 2022, 69(10): 4042-4055.](https://ieeexplore.ieee.org/abstract/document/9803862/) | Tsinghua University | -- |

## 4. Important Surveys on Edge AI (Related to edge inference and model deploy)
- [Convergence of edge computing and deep learning: A comprehensive survey. IEEE Communications Surveys & Tutorials, 22(2), 869-904.](https://ieeexplore.ieee.org/abstract/document/8976180/)
![](https://github.com/wangxb96/Awesome-Edge-Efficient-AI/blob/main/relation_ei_ie.png)
- [Deep learning with edge computing: A review. Proceedings of the IEEE, 107(8), 1655-1674.](https://ieeexplore.ieee.org/abstract/document/8763885/)
![](https://github.com/wangxb96/Awesome-Edge-Efficient-AI/blob/main/DNN_inference_speedup_methods.png)
- [Machine learning at the network edge: A survey. ACM Computing Surveys (CSUR), 54(8), 1-37.](https://dl.acm.org/doi/abs/10.1145/3469029)
![](https://github.com/wangxb96/Awesome-Edge-Efficient-AI/blob/main/ML_at_edge.png)
- [Edge intelligence: The confluence of edge computing and artificial intelligence. IEEE Internet of Things Journal, 7(8), 7457-7469.](https://ieeexplore.ieee.org/abstract/document/9052677/)
![](https://github.com/wangxb96/Awesome-Edge-Efficient-AI/blob/main/research_roadmap_edge_intelligence.png)
- [Edge intelligence: Paving the last mile of artificial intelligence with edge computing. Proceedings of the IEEE, 107(8), 1738-1762.](https://ieeexplore.ieee.org/abstract/document/8736011/)
![](https://github.com/wangxb96/Awesome-Edge-Efficient-AI/blob/main/six_level_EI.png)
- [Edge intelligence: Empowering intelligence to the edge of network. Proceedings of the IEEE, 109(11), 1778-1837.](https://ieeexplore.ieee.org/abstract/document/9596610/)
![](https://github.com/wangxb96/Awesome-Edge-Efficient-AI/blob/main/classification_of_edge_ai.png)

<!-- 
## 2. Papers 

### 2.1. Edge Computing
- [Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge computing: Vision and challenges. IEEE internet of things journal, 3(5), 637-646.](https://ieeexplore.ieee.org/abstract/document/7488250)
- [Varghese, B., Wang, N., Barbhuiya, S., Kilpatrick, P., & Nikolopoulos, D. S. (2016, November). Challenges and opportunities in edge computing. In 2016 IEEE International Conference on Smart Cloud (SmartCloud) (pp. 20-26). IEEE.](https://ieeexplore.ieee.org/abstract/document/7796149)
- [Shi, W., & Dustdar, S. (2016). The promise of edge computing. Computer, 49(5), 78-81.](https://ieeexplore.ieee.org/abstract/document/7469991)
- [Satyanarayanan, M. (2017). The emergence of edge computing. Computer, 50(1), 30-39.](https://ieeexplore.ieee.org/abstract/document/7807196)
- [Khan, W. Z., Ahmed, E., Hakak, S., Yaqoob, I., & Ahmed, A. (2019). Edge computing: A survey. Future Generation Computer Systems, 97, 219-235.](https://www.sciencedirect.com/science/article/pii/S0167739X18319903)
- [Abbas, N., Zhang, Y., Taherkordi, A., & Skeie, T. (2017). Mobile edge computing: A survey. IEEE Internet of Things Journal, 5(1), 450-465.](https://ieeexplore.ieee.org/abstract/document/8030322)
- [Mao, Y., You, C., Zhang, J., Huang, K., & Letaief, K. B. (2017). A survey on mobile edge computing: The communication perspective. IEEE communications surveys & tutorials, 19(4), 2322-2358.](https://ieeexplore.ieee.org/abstract/document/8016573)
- [Liu, F., Tang, G., Li, Y., Cai, Z., Zhang, X., & Zhou, T. (2019). A survey on edge computing systems and tools. Proceedings of the IEEE, 107(8), 1537-1562.](https://ieeexplore.ieee.org/abstract/document/8746691/)
- [Premsankar, G., Di Francesco, M., & Taleb, T. (2018). Edge computing for the Internet of Things: A case study. IEEE Internet of Things Journal, 5(2), 1275-1284.](https://ieeexplore.ieee.org/abstract/document/8289317/)
- [Xiao, Y., Jia, Y., Liu, C., Cheng, X., Yu, J., & Lv, W. (2019). Edge computing security: State of the art and challenges. Proceedings of the IEEE, 107(8), 1608-1631.](https://ieeexplore.ieee.org/abstract/document/8741060/)
- [Sonmez, C., Ozgovde, A., & Ersoy, C. (2018). Edgecloudsim: An environment for performance evaluation of edge computing systems. Transactions on Emerging Telecommunications Technologies, 29(11), e3493.](https://onlinelibrary.wiley.com/doi/abs/10.1002/ett.3493)
- [Li, H., Ota, K., & Dong, M. (2018). Learning IoT in edge: Deep learning for the Internet of Things with edge computing. IEEE network, 32(1), 96-101.](https://ieeexplore.ieee.org/abstract/document/8270639)
- [Hassan, N., Gillani, S., Ahmed, E., Yaqoob, I., & Imran, M. (2018). The role of edge computing in internet of things. IEEE communications magazine, 56(11), 110-115.](https://ieeexplore.ieee.org/abstract/document/8450541)
- [Sun, X., & Ansari, N. (2016). EdgeIoT: Mobile edge computing for the Internet of Things. IEEE Communications Magazine, 54(12), 22-29.](https://ieeexplore.ieee.org/abstract/document/7786106)
- [Pan, J., & McElhannon, J. (2017). Future edge cloud and edge computing for internet of things applications. IEEE Internet of Things Journal, 5(1), 439-449.](https://ieeexplore.ieee.org/abstract/document/8089336)
- [Liu, S., Liu, L., Tang, J., Yu, B., Wang, Y., & Shi, W. (2019). Edge computing for autonomous driving: Opportunities and challenges. Proceedings of the IEEE, 107(8), 1697-1716.](https://ieeexplore.ieee.org/abstract/document/8744265/)
- [Shi, W., Pallis, G., & Xu, Z. (2019). Edge computing [scanning the issue]. Proceedings of the IEEE, 107(8), 1474-1481.](https://ieeexplore.ieee.org/abstract/document/8789742/)
- [Chen, B., Wan, J., Celesti, A., Li, D., Abbas, H., & Zhang, Q. (2018). Edge computing in IoT-based manufacturing. IEEE Communications Magazine, 56(9), 103-109.](https://ieeexplore.ieee.org/abstract/document/8466364/)
- [Xiong, Z., Zhang, Y., Niyato, D., Wang, P., & Han, Z. (2018). When mobile blockchain meets edge computing. IEEE Communications Magazine, 56(8), 33-39.](https://ieeexplore.ieee.org/abstract/document/8436042)
- [Porambage, P., Okwuibe, J., Liyanage, M., Ylianttila, M., & Taleb, T. (2018). Survey on multi-access edge computing for internet of things realization. IEEE Communications Surveys & Tutorials, 20(4), 2961-2991.](https://ieeexplore.ieee.org/abstract/document/8391395/)
- [Ahmed, E., Ahmed, A., Yaqoob, I., Shuja, J., Gani, A., Imran, M., & Shoaib, M. (2017). Bringing computation closer toward the user network: Is edge computing the solution?. IEEE Communications Magazine, 55(11), 138-144.](https://ieeexplore.ieee.org/abstract/document/8114564/)
- [Taleb, T., Dutta, S., Ksentini, A., Iqbal, M., & Flinck, H. (2017). Mobile edge computing potential in making cities smarter. IEEE Communications Magazine, 55(3), 38-43.](https://ieeexplore.ieee.org/abstract/document/7876955/)
- [He, Q., Cui, G., Zhang, X., Chen, F., Deng, S., Jin, H., ... & Yang, Y. (2019). A game-theoretical approach for user allocation in edge computing environment. IEEE Transactions on Parallel and Distributed Systems, 31(3), 515-529.](https://ieeexplore.ieee.org/abstract/document/8823046/)
- [Mach, P., & Becvar, Z. (2017). Mobile edge computing: A survey on architecture and computation offloading. IEEE Communications Surveys & Tutorials, 19(3), 1628-1656.](https://ieeexplore.ieee.org/abstract/document/7879258/)
- [Tran, T. X., Hajisami, A., Pandey, P., & Pompili, D. (2017). Collaborative mobile edge computing in 5G networks: New paradigms, scenarios, and challenges. IEEE Communications Magazine, 55(4), 54-61.](https://ieeexplore.ieee.org/abstract/document/7901477)
- [Lin, L., Liao, X., Jin, H., & Li, P. (2019). Computation offloading toward edge computing. Proceedings of the IEEE, 107(8), 1584-1607.](https://ieeexplore.ieee.org/abstract/document/8758310/)
- [Qiu, T., Chi, J., Zhou, X., Ning, Z., Atiquzzaman, M., & Wu, D. O. (2020). Edge computing in industrial internet of things: Architecture, advances and challenges. IEEE Communications Surveys & Tutorials, 22(4), 2462-2488.](https://ieeexplore.ieee.org/abstract/document/9139976/)
- [Luo, Q., Hu, S., Li, C., Li, G., & Shi, W. (2021). Resource scheduling in edge computing: A survey. IEEE Communications Surveys & Tutorials, 23(4), 2131-2165.](https://ieeexplore.ieee.org/document/9519636?utm_source=researcher_app&utm_medium=referral&utm_campaign=RESR_MRKT_Researcher_inbound)
- [Taleb, T., Samdanis, K., Mada, B., Flinck, H., Dutta, S., & Sabella, D. (2017). On multi-access edge computing: A survey of the emerging 5G network edge cloud architecture and orchestration. IEEE Communications Surveys & Tutorials, 19(3), 1657-1681.](https://ieeexplore.ieee.org/abstract/document/7931566/)
- [Khan, L. U., Yaqoob, I., Tran, N. H., Kazmi, S. A., Dang, T. N., & Hong, C. S. (2020). Edge-computing-enabled smart cities: A comprehensive survey. IEEE Internet of Things Journal, 7(10), 10200-10232.](https://ieeexplore.ieee.org/abstract/document/9063670/)
- [Chen, X., Shi, Q., Yang, L., & Xu, J. (2018). ThriftyEdge: Resource-efficient edge computing for intelligent IoT applications. IEEE network, 32(1), 61-65.](https://ieeexplore.ieee.org/abstract/document/8270633/)
- [Baktir, A. C., Ozgovde, A., & Ersoy, C. (2017). How can edge computing benefit from software-defined networking: A survey, use cases, and future directions. IEEE Communications Surveys & Tutorials, 19(4), 2359-2391.](https://ieeexplore.ieee.org/abstract/document/7954011/)
- [Zhang, Z., Zhang, W., & Tseng, F. H. (2019). Satellite mobile edge computing: Improving QoS of high-speed satellite-terrestrial networks using edge computing techniques. IEEE network, 33(1), 70-76.](https://ieeexplore.ieee.org/abstract/document/8610431/)
- [Abdellatif, A. A., Mohamed, A., Chiasserini, C. F., Tlili, M., & Erbad, A. (2019). Edge computing for smart health: Context-aware approaches, opportunities, and challenges. IEEE Network, 33(3), 196-203.](https://ieeexplore.ieee.org/abstract/document/8674240/)
- [Liu, Y., Yang, C., Jiang, L., Xie, S., & Zhang, Y. (2019). Intelligent edge computing for IoT-based energy management in smart cities. IEEE network, 33(2), 111-117.](https://ieeexplore.ieee.org/abstract/document/8675180/)


### 2.2. Edge AI
- [Chen, J., & Ran, X. (2019). Deep learning with edge computing: A review. Proceedings of the IEEE, 107(8), 1655-1674.](https://ieeexplore.ieee.org/abstract/document/8763885/)
- [Deng, S., Zhao, H., Fang, W., Yin, J., Dustdar, S., & Zomaya, A. Y. (2020). Edge intelligence: The confluence of edge computing and artificial intelligence. IEEE Internet of Things Journal, 7(8), 7457-7469.](https://ieeexplore.ieee.org/abstract/document/9052677)
- [Zhou, Z., Chen, X., Li, E., Zeng, L., Luo, K., & Zhang, J. (2019). Edge intelligence: Paving the last mile of artificial intelligence with edge computing. Proceedings of the IEEE, 107(8), 1738-1762.](https://ieeexplore.ieee.org/abstract/document/8736011/)
- [Liu, Y., Peng, M., Shou, G., Chen, Y., & Chen, S. (2020). Toward edge intelligence: Multiaccess edge computing for 5G and Internet of Things. IEEE Internet of Things Journal, 7(8), 6722-6747.](https://ieeexplore.ieee.org/abstract/document/9123504)
- [Sodhro, A. H., Pirbhulal, S., & De Albuquerque, V. H. C. (2019). Artificial intelligence-driven mechanism for edge computing-based industrial applications. IEEE Transactions on Industrial Informatics, 15(7), 4235-4243.](https://ieeexplore.ieee.org/abstract/document/8658105/)
- [Li, E., Zeng, L., Zhou, Z., & Chen, X. (2019). Edge AI: On-demand accelerating deep neural network inference via edge computing. IEEE Transactions on Wireless Communications, 19(1), 447-457.](https://ieeexplore.ieee.org/abstract/document/8876870/)
- [Wang, X., Han, Y., Wang, C., Zhao, Q., Chen, X., & Chen, M. (2019). In-edge ai: Intelligentizing mobile edge computing, caching and communication by federated learning. IEEE Network, 33(5), 156-165.](https://ieeexplore.ieee.org/abstract/document/8770530/)
- [Xu, D., Li, T., Li, Y., Su, X., Tarkoma, S., Jiang, T., ... & Hui, P. (2020). Edge intelligence: Architectures, challenges, and applications. arXiv preprint arXiv:2003.12172.](https://arxiv.org/abs/2003.12172)
- [Li, E., Zhou, Z., & Chen, X. (2018, August). Edge intelligence: On-demand deep learning model co-inference with device-edge synergy. In Proceedings of the 2018 Workshop on Mobile Edge Communications (pp. 31-36).](https://dl.acm.org/doi/abs/10.1145/3229556.3229562)
- [Zhang, J., & Letaief, K. B. (2019). Mobile edge intelligence and computing for the internet of vehicles. Proceedings of the IEEE, 108(2), 246-261.](https://ieeexplore.ieee.org/abstract/document/8884164/)
- [Xiao, Y., Shi, G., Li, Y., Saad, W., & Poor, H. V. (2020). Toward self-learning edge intelligence in 6G. IEEE Communications Magazine, 58(12), 34-40.](https://ieeexplore.ieee.org/abstract/document/9311932/)
- [Zhang, K., Zhu, Y., Maharjan, S., & Zhang, Y. (2019). Edge intelligence and blockchain empowered 5G beyond for the industrial Internet of Things. IEEE network, 33(5), 12-19.](https://ieeexplore.ieee.org/abstract/document/8863721/)
- [Zhang, Y., Ma, X., Zhang, J., Hossain, M. S., Muhammad, G., & Amin, S. U. (2019). Edge intelligence in the cognitive Internet of Things: Improving sensitivity and interactivity. IEEE Network, 33(3), 58-64.](https://ieeexplore.ieee.org/abstract/document/8726073/)
- [Zhang, Y., Huang, H., Yang, L. X., Xiang, Y., & Li, M. (2019). Serious challenges and potential solutions for the industrial Internet of Things with edge intelligence. IEEE Network, 33(5), 41-45.](https://ieeexplore.ieee.org/abstract/document/8863725/)
- [Tang, H., Li, D., Wan, J., Imran, M., & Shoaib, M. (2019). A reconfigurable method for intelligent manufacturing based on industrial cloud and edge intelligence. IEEE Internet of Things Journal, 7(5), 4248-4259.](https://ieeexplore.ieee.org/abstract/document/8887246/)
- [Mills, J., Hu, J., & Min, G. (2019). Communication-efficient federated learning for wireless edge intelligence in IoT. IEEE Internet of Things Journal, 7(7), 5986-5994.](https://ieeexplore.ieee.org/abstract/document/8917724/)
- [Lim, W. Y. B., Ng, J. S., Xiong, Z., Jin, J., Zhang, Y., Niyato, D., ... & Miao, C. (2021). Decentralized edge intelligence: A dynamic resource allocation framework for hierarchical federated learning. IEEE Transactions on Parallel and Distributed Systems, 33(3), 536-550.](https://ieeexplore.ieee.org/abstract/document/9479786/)
- [Muhammad, K., Khan, S., Palade, V., Mehmood, I., & De Albuquerque, V. H. C. (2019). Edge intelligence-assisted smoke detection in foggy surveillance environments. IEEE Transactions on Industrial Informatics, 16(2), 1067-1075.](https://ieeexplore.ieee.org/abstract/document/8709763/)
- [Su, X., Sperlì, G., Moscato, V., Picariello, A., Esposito, C., & Choi, C. (2019). An edge intelligence empowered recommender system enabling cultural heritage applications. IEEE Transactions on Industrial Informatics, 15(7), 4266-4275.](https://ieeexplore.ieee.org/abstract/document/8675979/)
- [Yang, B., Cao, X., Xiong, K., Yuen, C., Guan, Y. L., Leng, S., ... & Han, Z. (2021). Edge intelligence for autonomous driving in 6G wireless system: Design challenges and solutions. IEEE Wireless Communications, 28(2), 40-47.](https://ieeexplore.ieee.org/abstract/document/9430907/)
- [Dai, Y., Zhang, K., Maharjan, S., & Zhang, Y. (2020). Edge intelligence for energy-efficient computation offloading and resource allocation in 5G beyond. IEEE Transactions on Vehicular Technology, 69(10), 12175-12186.](https://ieeexplore.ieee.org/abstract/document/9158401/)
- [Al-Rakhami, M., Gumaei, A., Alsahli, M., Hassan, M. M., Alamri, A., Guerrieri, A., & Fortino, G. (2020). A lightweight and cost effective edge intelligence architecture based on containerization technology. World Wide Web, 23(2), 1341-1360.](https://link.springer.com/article/10.1007/s11280-019-00692-y)
- [Feng, C., Yu, K., Aloqaily, M., Alazab, M., Lv, Z., & Mumtaz, S. (2020). Attribute-based encryption with parallel outsourced decryption for edge intelligent IoV. IEEE Transactions on Vehicular Technology, 69(11), 13784-13795.](https://ieeexplore.ieee.org/abstract/document/8863733/)
- [Feng, C., Yu, K., Aloqaily, M., Alazab, M., Lv, Z., & Mumtaz, S. (2020). Attribute-based encryption with parallel outsourced decryption for edge intelligent IoV. IEEE Transactions on Vehicular Technology, 69(11), 13784-13795.](https://ieeexplore.ieee.org/abstract/document/9209123/)
- [Chen, B., Wan, J., Lan, Y., Imran, M., Li, D., & Guizani, N. (2019). Improving cognitive ability of edge intelligent IIoT through machine learning. IEEE network, 33(5), 61-67.](https://ieeexplore.ieee.org/abstract/document/8863728/)
- [Park, J., Samarakoon, S., Bennis, M., & Debbah, M. (2019). Wireless network intelligence at the edge. Proceedings of the IEEE, 107(11), 2204-2239.](https://ieeexplore.ieee.org/abstract/document/8865093/)
- [Tang, S., Chen, L., He, K., Xia, J., Fan, L., & Nallanathan, A. (2022). Computational intelligence and deep learning for next-generation edge-enabled industrial IoT. IEEE Transactions on Network Science and Engineering.](https://ieeexplore.ieee.org/abstract/document/9790341/)
- [Zhang, W., Zhang, Z., Zeadally, S., Chao, H. C., & Leung, V. C. (2019). MASM: A multiple-algorithm service model for energy-delay optimization in edge artificial intelligence. IEEE Transactions on Industrial Informatics, 15(7), 4216-4224.](https://ieeexplore.ieee.org/abstract/document/8632751/)
- [Ghosh, A. M., & Grolinger, K. (2020). Edge-cloud computing for internet of things data analytics: embedding intelligence in the edge with deep learning. IEEE Transactions on Industrial Informatics, 17(3), 2191-2200.](https://ieeexplore.ieee.org/abstract/document/9139356/)
- [Kang, Y., Hauswald, J., Gao, C., Rovinski, A., Mudge, T., Mars, J., & Tang, L. (2017). Neurosurgeon: Collaborative intelligence between the cloud and mobile edge. ACM SIGARCH Computer Architecture News, 45(1), 615-629.](https://dl.acm.org/doi/abs/10.1145/3093337.3037698)
- [Zhu, G., Liu, D., Du, Y., You, C., Zhang, J., & Huang, K. (2020). Toward an intelligent edge: Wireless communication meets machine learning. IEEE communications magazine, 58(1), 19-25.](https://ieeexplore.ieee.org/abstract/document/8970161/)
- [Yang, H., Wen, J., Wu, X., He, L., & Mumtaz, S. (2019). An efficient edge artificial intelligence multipedestrian tracking method with rank constraint. IEEE Transactions on Industrial Informatics, 15(7), 4178-4188.](https://ieeexplore.ieee.org/abstract/document/8633399/)
- [Shi, Y., Yang, K., Jiang, T., Zhang, J., & Letaief, K. B. (2020). Communication-efficient edge AI: Algorithms and systems. IEEE Communications Surveys & Tutorials, 22(4), 2167-2191.](https://ieeexplore.ieee.org/abstract/document/9134426/)
- [Ding, A. Y., Peltonen, E., Meuser, T., Aral, A., Becker, C., Dustdar, S., ... & Wolf, L. (2022). Roadmap for edge AI: a Dagstuhl perspective. ACM SIGCOMM Computer Communication Review, 52(1), 28-33.](https://dl.acm.org/doi/abs/10.1145/3523230.3523235)
- [Soro, S. (2021). TinyML for ubiquitous edge AI. arXiv preprint arXiv:2102.01255.](https://arxiv.org/ftp/arxiv/papers/2102/2102.01255.pdf)
- [Letaief, K. B., Shi, Y., Lu, J., & Lu, J. (2021). Edge artificial intelligence for 6G: Vision, enabling technologies, and applications. IEEE Journal on Selected Areas in Communications, 40(1), 5-36.](https://ieeexplore.ieee.org/abstract/document/9606720/)
- [Ke, R., Zhuang, Y., Pu, Z., & Wang, Y. (2020). A smart, efficient, and reliable parking surveillance system with edge artificial intelligence on IoT devices. IEEE Transactions on Intelligent Transportation Systems, 22(8), 4962-4974.](https://ieeexplore.ieee.org/abstract/document/9061155/)
- [Marculescu, R., Marculescu, D., & Ogras, U. (2020, August). Edge AI: Systems design and ML for IoT data analytics. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 3565-3566).](https://dl.acm.org/doi/abs/10.1145/3394486.3406479)
- [Mahendran, J. K., Barry, D. T., Nivedha, A. K., & Bhandarkar, S. M. (2021). Computer vision-based assistance system for the visually impaired using mobile edge artificial intelligence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2418-2427).](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Mahendran_Computer_Vision-Based_Assistance_System_for_the_Visually_Impaired_Using_Mobile_CVPRW_2021_paper.html)
- [Stäcker, L., Fei, J., Heidenreich, P., Bonarens, F., Rambach, J., Stricker, D., & Stiller, C. (2021). Deployment of Deep Neural Networks for Object Detection on Edge AI Devices with Runtime Optimization. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1015-1022).](https://openaccess.thecvf.com/content/ICCV2021W/ERCVAD/html/Stacker_Deployment_of_Deep_Neural_Networks_for_Object_Detection_on_Edge_ICCVW_2021_paper.html)
- [Zawish, M., Davy, S., & Abraham, L. (2022). Complexity-driven cnn compression for resource-constrained edge ai. arXiv preprint arXiv:2208.12816.](https://arxiv.org/abs/2208.12816)
- [Yao, J., Zhang, S., Yao, Y., Wang, F., Ma, J., Zhang, J., ... & Yang, H. (2022). Edge-Cloud Polarization and Collaboration: A Comprehensive Survey for AI. IEEE Transactions on Knowledge and Data Engineering.](https://ieeexplore.ieee.org/abstract/document/9783185/)
-->
