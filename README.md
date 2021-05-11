# Multi-Modal-Dialgoue-System-Paperlist

This is a paper list for the multimodal dialogue systems topic.

**Keyword**: Multi-modal, Dialogue system, visual, conversation

# Paperlist

## Dataset & Challenges

### Images
 
(1) [**Visual QA**](https://visualqa.org/workshop.html) VQA datasets in CVPR2021,2020,2019,..., containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
- [VQA](https://visualqa.org/challenge) datasets [1.0](http://arxiv.org/abs/1505.00468) [2.0](https://arxiv.org/abs/1612.00837)
- [TextVQA](https://textvqa.org/paper) TextVQA requires models to read and reason about text in an image to answer questions based on them. In order to perform well on this task, models need to first detect and read text in the images. Models then need to reason about this to answer the question. 
- [TextCap](https://arxiv.org/abs/2003.12462) TextCaps requires models to read and reason about text in images to generate captions about them. Specifically, models need to incorporate a new modality of text present in the images and reason over it and visual content in the image to generate image descriptions.
- Issues : 
  - visual-explainable: the model should rely on the right visual regions when making decisions, 
  - question-sensitive: the model should be sensitive to the linguistic variations in question
  - reduce language biases: the model should not take the language shortcut to answer the question without looking at the image
- Further Papers (too many)
  - cross-modal interaction /fusion
    - [Multimodal Neural Graph Memory Networks for Visual Question Answering](https://www.aclweb.org/anthology/2020.acl-main.643.pdf) ACL2020
    - [Bottom-up and top-down attention for image captioning and visual question answering](https://arxiv.org/abs/1707.07998) in CVPR2018, winner of the 2017 Visual Question Answering challenge
    - [Multimodal Neural Graph Memory Networks for Visual Question Answering](https://www.aclweb.org/anthology/2020.acl-main.643.pdf) ACL2020, visual features + encoded region-grounded captions (of object attributes and their relationships) = two graph nets which compute question-guided contextualized representation for each, then the updated representations are written to an external spatial memory (??what's that??).
    - [Cross-Modality Relevance for Reasoning on Language and Vision](https://www.aclweb.org/anthology/2020.acl-main.683.pdf) in ACL2020
    - [Hypergraph Attention Networks for Multimodal Learning](https://bi.snu.ac.kr/~btzhang/selected_papers/CVPR2020_ESKimKOHZ.pdf) CVPR2020
    - [Human Attention in Visual Question Answering: Do Humans and Deep Networks look at the same regions?] EMNLP2016
    - [Multi-level Attention Networks for Visual Question Answering] CVPR2017
    - [Hierarchical Question-Image Co-Attention for Visual Question Answering] CVPR2016
  - vision-language pretraining / representation learning
    - [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557.pdf) arXiv2019, ground element of language to image regions with self-attention
    - [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks] NeuIPS2019
    - [VL-BERT: Pre-training of Generic Visual-Linguistic Representations] ICRL2020
    - [VinVL: Making Visual Representations Matter in Vision-Language Models]
    - [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision] arXiv 2021
    - [12-in-1: Multi-Task Vision and Language Representation Learning] CVPR2020
    - [Unified Vision-Language Pre-Training for Image Captioning and VQA] AAAI2020
    - [LXMERT: Learning Cross-Modality Encoder Representations from Transformers] EMNLP2019
    - [Adaptive Transformers for Learning Multimodal Representations](https://www.aclweb.org/anthology/2020.acl-srw.1.pdf) ACL2020
    - [Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer](https://www.aclweb.org/anthology/2020.acl-main.306.pdf) ACL2020
  - Language prior issue
    - [AdaVQA: Overcoming Language Priors with Adapted Margin Cosine Loss](https://arxiv.org/pdf/2105.01993.pdf) in a perspective of feature space learning (not classification task)
    - [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837) CVPR2017 VQA 2.0 is also for the purpose of balance language prior to images 
    - [Self-Critical Reasoning for Robust Visual Question Answering](https://arxiv.org/pdf/1905.09998.pdf) NeurIPS2019
    - [Overcoming Language Priors in Visual Question Answering with Adversarial Regularization](https://arxiv.org/pdf/1810.03649.pdf) NeurIPS2018, question-only model
    - [RUBi: Reducing Unimodal Biases in Visual Question Answering](https://arxiv.org/pdf/1906.10169.pdf) NeurIPS2019 also question-only model
    - [Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering] CVPR2018
    - [Counterfactual VQA: A Cause-Effect Look at Language Bias] 2020
    - [Counterfactual Vision and Language Learning] CVPR2020
  - Visual-explainable issue
    - [Counterfactual Samples Synthesizing for Robust Visual Question Answering](http://arxiv.org/pdf/2003.06576) CVPR2020
    - [Learning to Contrast the Counterfactual Samples for Robust Visual Question Answering](https://www.aclweb.org/anthology/2020.emnlp-main.265.pdf) EMNLP2020
    - [Learning What Makes a Difference from Counterfactual Examples and Gradient Supervision](https://arxiv.org/pdf/2004.09034.pdf) ECCV2020 leveraging overlooked supervisory signal found in existing datasets to improve generalization capabilities
    - [Generating Natural Language Explanations for Visual Question Answering using Scene Graphs and Visual Attention] arXiv2019
    - [Towards Transparent AI Systems: Interpreting Visual Question Answering Models] 2016
  - object relation reasoning / visual understanding / cross-modal / Graphs
    - [MUREL: Multimodal Relational Reasoning for Visual Question Answering](http://arxiv.org/pdf/1902.09487) CVPR2019, [[Code](github.com/Cadene/murel.bootstrap.pytorch)], represent and refine interactions between question words and image regions, more fine than attention-maps
    - [CRA-Net: Composed Relation Attention Network for Visual Question Answering](https://dl.acm.org/doi/10.1145/3343031.3350925) ACM2019 object relation reasoning attention should look at both visual (features, spatial) and linguistic (in questions) features 不让看哦？
    - [Hierarchical Graph Attention Network for Visual Relationship Detection](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mi_Hierarchical_Graph_Attention_Network_for_Visual_Relationship_Detection_CVPR_2020_paper.pdf) CVPR2020 object-level graph: (1) woman (sit on) bench, (2) woman (in front of) water; triplet-level graph: relation between triplet(1) and triplet(2)
    - [Visual Relationship Detection With Visual-Linguistic Knowledge From Multimodal Representations](https://ieeexplore.ieee.org/ielx7/6287639/6514899/09387302.pdf) IEEE2021, relational visual-linguistic BERT
    - [Relation-Aware Graph Attention Network for Visual Question Answering](http://arxiv.org/pdf/1903.12314) ICCV2019, explicit relations of geometric positions and semantic interactions between objects, implicit relations of hidden dynamics between image regions
    - [Fusion of Detected Objects in Text for Visual Question Answering] EMNLP2020
    - [GraghVQA: Language-Guided Graph Neural Networks for Graph-based Visual Question Answering] arXiv2021
    - [A Simple Baseline for Visual Commonsense Reasoning] ViGil@NeuIPS2019
    - [Learning Conditioned Graph Structures for Interpretable Visual Question Answering] NeuIPS2018
    - [Graph-Structured Representations for Visual Question Answering] CVPR2017
    - [R-VQA: Learning Visual Relation Facts with Semantic Attention for Visual Question Answering] ACM 2018
  - Knowledge / cross-modal fusion / Graphs
    - [Towards Knowledge-Augmented Visual Question Answering](https://www.aclweb.org/anthology/2020.coling-main.169.pdf) Coling2020, capture the interactions between objects in a visual scene and entities in an external knowledge source, with many many graphs ...
    - [ConceptBert: Concept-Aware Representation for Visual Question Answering](https://www.aclweb.org/anthology/2020.findings-emnlp.44.pdf) EMNLP2020, learn a joint Concept-Vision-Language embedding (maybe similar to [[this paper](https://openreview.net/references/pdf?id=Uhl6chXANP)] in the way of adding "entity embedding" ?)
    - [Incorporating External Knowledge to Answer Open-Domain Visual Questions with Dynamic Memory Networks] 2017
  - text in the image (TextCap & TextVQA)
    - [Multi-Modal Graph Neural Network for Joint Reasoning on Vision and Scene Tex](http://arxiv.org/pdf/2003.13962) CVPR2020, the printed text on the bottle is the brand of the drink ==> graph representation of the image should have sub-graphs and respective aggregators to pass messages among graphs (我不知道我在说什么???)
    - [Multi-Modal Reasoning Graph for Scene-Text Based Fine-Grained Image Classification and Retrieval](https://arxiv.org/pdf/2009.09809.pdf) arXiv2020, common semantic space between salient objects and text found in an image
    - [Simple is not Easy: A Simple Strong Baseline for TextVQA and TextCaps](https://arxiv.org/pdf/2012.05153.pdf) arXiv2020, simple attention mechanism is, good 
    - [Cascade Reasoning Network for Text-based Visual Question Answering](https://tanmingkui.github.io/files/publications/Cascade.pdf) ACM2020, 1) which info's useful, 2)question related to text but also visual concepts, how to capture cross-modal relathionships, 3)what if OCR fails 
    - [TAP: Text-Aware Pre-training for Text-VQA and Text-Caption](https://arxiv.org/pdf/2012.04638.pdf) arXiv2020, incorporates OCR generated text in pre-training
    - [Iterative Answer Prediction With Pointer-Augmented Multimodal Transformers for TextVQA]
  - multi-task
    - [Answer Them All! Toward Universal Visual Question Answering Models] CVPR2019
    - [A Recipe for Creating Multimodal Aligned Datasets for Sequential Tasks](https://www.aclweb.org/anthology/2020.acl-main.440.pdf) ACL2020
    - [Visual Question Answering as a Multi-Task Problem](https://arxiv.org/pdf/2007.01780.pdf) arXiv2020
    

(2) [**Visual Dialog**](https://visualdialog.org/) CVPR 2017, Open-domain dialogs & given an image, a dialog history, and a follow-up question about the image, the task is to answer the question. 
- [VisDial v1.0 dataset](https://visualdialog.org/data) [[Paper](https://arxiv.org/abs/1611.08669)] [[Source Code to collect chat data](https://github.com/batra-mlp-lab/visdial-amt-chat)]
- Further papers
  - reasoning
    - [KBGN: Knowledge-Bridge Graph Network for Adaptive Vision-Text Reasoning in Visual Dialogue](http://arxiv.org/pdf/2008.04858) ACM2020, here knowledge = text knowledge & vision knowledge, encoding (T2V graph & V2T graph) then bridging (update graph nodes) then storing then retrieving (via adaptive information selection mode)
    - [Multi-step Reasoning via Recurrent Dual Attention for Visual Dialog](https://www.aclweb.org/anthology/P19-1648.pdf) ACL2019, iteratively refine the question's representation based on image and dialog history
    - [DMRM: A Dual-channel Multi-hop Reasoning Model for Visual Dialog] AAAI2020
    - [Visual Reasoning with Multi-hop Feature Modulation] ECCV2018
    - [VisualCOMET: Reasoning About the Dynamic Context of a Still Image] ECCV2020
  - understanding
    - [DualVD: An Adaptive Dual Encoding Model for Deep Visual Understanding in Visual Dialogue](https://arxiv.org/pdf/1911.07251.pdf)
    - [Learning Dual Encoding Model for Adaptive Visual Understanding in Visual Dialogue](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9247486) IEEE2021
    - [Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision] EMNLP2020   
  - coreference
    - [Modeling Coreference Relations in Visual Dialog] EACL2021
    - [What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues] EMNLP2019
    - [Dual Attention Networks for Visual Reference Resolution in Visual Dialog] EMNLP2019
  - cross-modal / fusion / joint / dual ...
    - [Efficient Attention Mechanism for Handling All the Interactions between Many Inputs with Application to Visual Dialog] 2019
    - [Image-Question-Answer Synergistic Network for Visual Dialog] CVPR2019
    - [DialGraph: Sparse Graph Learning Networks for Visual Dialog]
    - [All-in-One Image-Grounded Conversational Agents] arXiv2019
    - [Visual-Textual Alignment for Graph Inference in Visual Dialog] Coling2020
    - [Connecting Language and Vision to Actions] ACL2018
    - [Parallel Attention: A Unified Framework for Visual Object Discovery Through Dialogs and Queries] CVPR2018
    - [Neural Multimodal Belief Tracker with Adaptive Attention for Dialogue Systems] WWW2019
    - [Reactive Multi-Stage Feature Fusion for Multimodal Dialogue Modeling] 2019
    - [Two Causal Principles for Improving Visual Dialog] CVPR2020
    - [Learning Cross-modal Context Graph for Visual Grounding] AAAI2020
    - [Multi-View Attention Networks for Visual Dialog] arXiv2020
    - [Efficient Attention Mechanism for Visual Dialog that Can Handle All the Interactions Between Multiple Inputs] ECCV2020
    - [Shuffle-Then-Assemble: Learning Object-Agnostic Visual Relationship Features] ECCV2018
    - [Investigating topics, audio representations and attention for multimodal scene-aware dialog] 2020
  - referring / embodiment / grounding
    - [Where Are You? Localization from Embodied Dialog] EMNLP2020
    - [Visual Reference Resolution using Attention Memory for Visual Dialog] NIPS2017
    - [Referring Expression Generation via Visual Dialogue] NLPCC2020
    - [Visual Referring Expression Recognition: What Do Systems Actually Learn?] NAACL2018
    - [Ask No More: Deciding when to guess in referential visual dialogue] coling2018
    - [Refer, Reuse, Reduce: Generating Subsequent References in Visual and Conversational Contexts] EMNLP2020
    - [Situated and Interactive Multimodal Conversations] EMNLP2020
    - [Achieving Common Ground in Multi-modal Dialogue] ACL2020
  - use dialog history / user guided
    - [Making History Matter: History-Advantage Sequence Training for Visual Dialog] ICCV2019
    - [User Attention-guided Multimodal Dialog Systems] SIGIR2019
    - [History for Visual Dialog: Do we really need it?] ACL2020
    - [Integrating Historical States and Co-attention Mechanism for Visual Dialog] ICPR2021
  - knowledge
    - [The Dialogue Dodecathlon: Open-Domain Knowledge and Image Grounded Conversational Agents] ACL2020
    - [Knowledge-aware Multimodal Dialogue Systems] ACM2018
    - [A Knowledge-Grounded Multimodal Search-Based Conversational Agent] SCAI@EMNLP2018
  - modality bias
    - [Modality-Balanced Models for Visual Dialogue] AAAI2020
    - [Training data-efficient image transformers & distillation through attention] arXiv
    - [Unsupervised Natural Language Inference via Decoupled Multimodal Contrastive Learning] EMNLP2020
    - [Visual Dialogue without Vision or Dialogue] 2018
    - [Be Different to Be Better! A Benchmark to Leverage the Complementarity of Language and Vision] EMNLP2020
    - [Worst of Both Worlds: Biases Compound in Pre-trained Vision-and-Language Models] 2021
  - pretraining / representation learning / bertologie
    - [VD-BERT: A Unified Vision and Dialog Transformer with BERT] EMNLP2020
    - [Large-scale Pretraining for Visual Dialog: A Simple State-of-the-Art Baseline] ECCV2020
    - [Kaleido-BERT: Vision-Language Pre-training on Fashion Domain] arXiv2021
    - [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks] ECCV2020
    - [12-in-1: Multi-Task Vision and Language Representation Learning] CVPR2020
    - [Large-Scale Adversarial Training for Vision-and-Language Representation Learning] 2020
    - [Integrating Multimodal Information in Large Pretrained Transformers](https://www.aclweb.org/anthology/2020.acl-main.214.pdf) ACL2020
  - visual retrieval / image retrieval / 
    - [Thinking Fast and Slow: Efficient Text-to-Visual Retrieval with Transformers] 
    - [Exploring Phrase Grounding without Training: Contextualisation and Extension to Text-Based Image Retrieval] CVPRW2020
    - [Interactive Classification by Asking Informative Questions] ACL2020
    - [Toward General Scene Graph: Integration of Visual Semantic Knowledge with Entity Synset Alignment] ALVR2020
    - [Dialog-based Interactive Image Retrieval] NeuIPS2018
    - [I Want This Product but Different : Multimodal Retrieval with Synthetic Query Expansion] 2021
  - image editing / text-to-image
    - [Sequential Attention GAN for Interactive Image Editing] ACM2020
    - [Tell, Draw, and Repeat: Generating and Modifying Images Based on Continual Linguistic Instruction] ICCV2019
    - [ChatPainter: Improving Text to Image Generation using Dialogue] ICLR2018
    - [Adversarial Text-to-Image Synthesis: A Review] 2021
    - [A Multimodal Dialogue System for Conversational Image Editing] 2020
  - Generative dialogue / diverse
    - [Improving Generative Visual Dialog by Answering Diverse Questions](https://arxiv.org/pdf/1909.10470.pdf) EMNLP 2019, [[Code](https://github.com/vmurahari3/visdial-diversity)]
    - [Visual Dialogue State Tracking for Question Generation] AAAI2020
    - [MultiDM-GCN: Aspect-Guided Response Generation in Multi-Domain Multi-Modal Dialogue System using Graph Convolution Network] EMNLP2020
    - [Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model] NIPS2017
    - [FLIPDIAL: A Generative Model for Two-Way Visual Dialogue] CVPR2018
    - [DAM: Deliberation, Abandon and Memory Networks for Generating Detailed and Non-repetitive Responses in Visual Dialogue] IJCAI2020
    - [More to diverse: Generating diversified responses in a task oriented multimodal dialog system] 2020
    - [Multimodal Dialog System: Generating Responses via Adaptive Decoders] ACM2019
    - [Image-Grounded Conversations: Multimodal Context for Natural Question and Response Generation] IJCNLP2017
    - [Open Domain Dialogue Generation with Latent Images] 2020
    - [Multimodal Differential Network for Visual Question Generation] EMNLP2018
    - [Generative Visual Dialogue System via Adaptive Reasoning and Weighted Likelihood Estimation] 2019
    - [Aspect-Aware Response Generation for Multimodal Dialogue System] ACM2021
    - [An Empirical Study on the Generalization Power of Neural Representations Learned via Visual Guessing Games] EACL2021
  - Adversarial training
    - [The World in My Mind: Visual Dialog with Adversarial Multi-modal Feature Encoding] NAACL2019
    - [Mind Your Language: Learning Visually Grounded Dialog in a Multi-Agent Setting] 2018
    - [GADE: A Generative Adversarial Approach to Density Estimation and its Applications] IJCV2020
  - RL
    - [Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1703.06585) ICCV2017 oral, [[Code](https://github.com/batra-mlp-lab/visdial-rl)]
    - [Multimodal Hierarchical Reinforcement Learning Policy for Task-Oriented Visual Dialog] SIGDIAL 2018
    - [Multimodal Dialog for Browsing Large Visual Catalogs using Exploration-Exploitation Paradigm in a Joint Embedding Space] ICMR2019
    - [Recurrent Attention Network with Reinforced Generator for Visual Dialog] ACM 2020
  - linguistic / probabilistic
    - [A Linguistic Analysis of Visually Grounded Dialogues Based on Spatial Expressions] EMNLP2020
    - [Probabilistic framework for solving Visual Dialog] PR 2021
    - [Learning Goal-Oriented Visual Dialog Agents: Imitating and Surpassing Analytic Experts] IEEE2019
  - goal-oriented
    - [Answerer in Questioner's Mind for Goal-Oriented Visual Dialogue] arXiv2018
    - [End-to-end optimization of goal-driven and visually grounded dialogue systems] IJCAI2017
    - [Learning Goal-Oriented Visual Dialog via Tempered Policy Gradient] IEEE2018
    - [An Active Information Seeking Model for Goal-oriented Vision-and-Language Tasks] 2018
    - [Answer-Driven Visual State Estimator for Goal-Oriented Visual Dialogue] ACM 2020
    - [The Interplay of Task Success and Dialogue Quality: An in-depth Evaluation in Task-Oriented Visual Dialogues] EACL2021
    - [Building Task-Oriented Visual Dialog Systems Through Alternative Optimization Between Dialog Policy and Language Generation] EMNLP2019
    - [An Active Information Seeking Model for Goal-oriented Vision-and-Language Tasks] 2018
    - [Storyboarding of Recipes: Grounded Contextual Generation] DGS@ICLR2019
    - [Gold Seeker: Information Gain From Policy Distributions for Goal-Oriented Vision-and-Langauge Reasoning] CVPR2020
  - open-domain
    - [Multi-Modal Open-Domain Dialogue] 2020
    - [Image-Chat: Engaging Grounded Conversations] ACL2020
    - [The Dialogue Dodecathlon: Open-Domain Knowledge and Image Grounded Conversational Agents](https://www.aclweb.org/anthology/2020.acl-main.222.pdf) ACL2020
  - evaluation
    - [A Revised Generative Evaluation of Visual Dialogue] arXiv2020
    - [Evaluating Visual Conversational Agents via Cooperative Human-AI Games] 2017
    - [The Interplay of Task Success and Dialogue Quality: An in-depth Evaluation in Task-Oriented Visual Dialogues] EACL2021


(3) [CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog](https://www.aclweb.org/anthology/N19-1058) NAACL2019, [[code](https://github.com/satwikkottur/clevr-dialog)]
- Further paper
  - [VQA With No Questions-Answers Training](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9157617) CVPR2020
  - [Domain-robust VQA with diverse datasets and methods but no target labels](https://arxiv.org/pdf/2103.15974.pdf) arXiv2021
  - [Scene Graph based Image Retrieval - A case study on the CLEVR Dataset] 2019

(4) Open-domain:
- [OpenViDial: A Large-Scale, Open-Domain Dialogue Dataset with Visual Contexts]
- [The PhotoBook Dataset: Building Common Ground through Visually-Grounded Dialogue] ACL2019
- [A Visually-Grounded Parallel Corpus with Phrase-to-Region Linking] LREC2020

(?) sentiment
- [MEISD: A Multimodal Multi-Label Emotion, Intensity and Sentiment Dialogue Dataset for Emotion Recognition and Sentiment Analysis in Conversations] Coling2020
- [Bridging Dialogue Generation and Facial Expression Synthesis] 2019

(5) Task/Goal-oriented:
- [CRWIZ: A Framework for Crowdsourcing Real-Time Wizard-of-Oz Dialogues] LREC2020
- [A Corpus for Reasoning About Natural Language Grounded in Photographs] ACL2019
- [CoDraw: Collaborative Drawing as a Testbed for Grounded Goal-driven Communication] ACL2019
- [AirDialogue: An Environment for Goal-Oriented Dialogue Research] EMNLP2018
- [ReferIt](http://tamaraberg.com/referitgame/) [[paper](http://tamaraberg.com/papers/referit.pdf)] in EMNLP2014, 2-players game of refer & label


(6) Others
- [Fatality Killed the Cat or: BabelPic, a Multimodal Dataset for Non-Concrete Concepts](https://www.aclweb.org/anthology/2020.acl-main.425.pdf) ACL2020
- [How2: A Large-scale Dataset for Multimodal Language Understanding] NIPS2018

(7) Games
- [**GuessWhat?!**](https://openaccess.thecvf.com/content_cvpr_2017/html/de_Vries_GuessWhat_Visual_Object_CVPR_2017_paper.html) Visual Object Discovery Through Multi-Modal Dialogue in CVPR2017, a two-player guessing game (1 oracle & 1 questioner). 
  - [[Code]](https://github.com/GuessWhatGame/guesswhat)
  - Further paper 
    - [End-to-end optimization of goal-driven and visually grounded dialogue systems](https://arxiv.org/abs/1703.05423) Reinforcement Learning applied to GuessWhat?! 
    - [Guessing State Tracking for Visual Dialogue] ECCV2020
    - [Language-Conditioned Feature Pyramids for Visual Selection Tasks] EMNLP2020 [[Code](https://github.com/Alab-NII/lcfp)]
    - [Jointly Learning to See, Ask, and GuessWhat] arXiv2018
    - [Beyond task success: A closer look at jointly learning to see, ask, and GuessWhat] NAACL2019
- Minecraft
  - [Learning to execute instructions in a Minecraft dialogue] ACL2020
  - [Collaborative Dialogue in Minecraft] ACL2020
- [Friend, Collaborator, Student, Manager: How Design of an AI-Driven Game Level Editor Affects Creators] CHI2019



(8) [Image caption] generating natural language description of an image
- [MS COCO dataset 2014](https://arxiv.org/pdf/1405.0312.pdf) Images + captions (but captions are single words not sentences)
- Further papers
  - Feature images as a whole / and regions (early approachs) : 
    - [Deep visual-semantic alignments for generating image descriptions](https://cs.stanford.edu/people/karpathy/deepimagesent/devisagen.pdf) CVPR2015
    - [Densecap: Fully convolutional localization networks for dense captioning](https://cs.stanford.edu/people/karpathy/densecap/) CVPR2016
  - Attention based approaches :
    - [Bottom-up and top-down attention for image captioning and visual question answering](https://arxiv.org/abs/1707.07998) in CVPR2018, winner of the 2017 Visual Question Answering challenge
    - [Show, attend and tell: Neural image caption generation with visual attention](https://arxiv.org/abs/1502.03044) in ICML2015
    - [Review networks for caption generation](https://arxiv.org/abs/1605.07912) NIPS2016
    - [Image captioning with semantic attention](https://arxiv.org/abs/1603.03925) CVPR2016
  - Graph structured approaches :
    - [Exploring Visual Relationship for Image Captioning](https://arxiv.org/abs/1809.07041) ECCV2018
    - [Auto-encoding scene graphs for image captioning](https://arxiv.org/abs/1812.02378) CVPR2019
  - Reinforcement learning:
    - [Context-aware visual policy network for sequence-level image captioning](https://arxiv.org/abs/1808.05864) ACM2018
    - [Self-critical sequence training for image captioning](https://arxiv.org/abs/1612.00563)
  - Transformer based:
    - [Image captioning: transform objects into words](https://papers.nips.cc/paper/2019/file/680390c55bbd9ce416d1d69a9ab4760d-Paper.pdf) in NIPS2019 using Transformers focusing on objects and their spatial relationships
    - [Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning](https://www.aclweb.org/anthology/P18-1238.pdf) in ACL2018, also a dataset
    - [Improving Image Captioning with Better Use of Caption](https://www.aclweb.org/anthology/2020.acl-main.664.pdf) ACL2020
    - [Improving Image Captioning Evaluation by Considering Inter References Variance](https://www.aclweb.org/anthology/2020.acl-main.93.pdf) ACL2020


(9) Navigation task
- [Talk the walk: Navigating new york city through grounded dialogue](https://arxiv.org/pdf/1807.03367.pdf)
- [A Visually-grounded First-person Dialogue Dataset with Verbal and Non-verbal Responses] EMNLP2020
  - navigating
    - [Improving Vision-and-Language Navigation with Image-Text Pairs from the Web] ECCV2020
    - [Diagnosing Vision-and-Language Navigation: What Really Matters] arXiv2021
    - [Vision-Dialog Navigation by Exploring Cross-Modal Memory] CVPR2020
    - [Vision-and-Dialog Navigation] CoVR 2019
    - [Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments] CVPR2018
    - [Embodied Vision-and-Language Navigation with Dynamic Convolutional Filters] 2020
    - [Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation] ACL2019
    - [Active Visual Information Gathering for Vision-Language Navigation] ECCV2020
    - [Environment-agnostic Multitask Learning for Natural Language Grounded Navigation] ECCV2020
    - [Perceive, Transform, and Act: Multi-Modal Attention Networks for Vision-and-Language Navigation] 2019
    - [Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation] CVPR2019
    - [Engaging Image Chat: Modeling Personality in Grounded Dialogue] 2018
    - [TOUCHDOWN: Natural Language Navigation and Spatial Reasoning in Visual Street Environments] CVPR2019
    - [Multi-modal Discriminative Model for Vision-and-Language Navigation] 2019
    - [REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments] CVPR 2020
    - [Learning To Follow Directions in Street View] AAAI2020
    - [Help, Anna! Visual Navigation with Natural Multimodal Assistance via Retrospective Curiosity-Encouraging Imitation Learning] ViGil@NeuIPS2019
  - representation learning
    - [A Recurrent Vision-and-Language BERT for Navigation] arXiv2020
    - [Transferable Representation Learning in Vision-and-Language Navigation] ICCV2019
  - Grounding
    - [Words Aren't Enough, Their Order Matters: On the Robustness of Grounding Visual Referring Expressions](https://www.aclweb.org/anthology/2020.acl-main.586.pdf) ACL2020
    - [Grounding Conversations with Improvised Dialogues](https://www.aclweb.org/anthology/2020.acl-main.218.pdf) ACL2020
    - [A negative case analysis of visual grounding methods for VQA](https://www.aclweb.org/anthology/2020.acl-main.727.pdf) ACL2020
    - [Knowledge Supports Visual Language Grounding: A Case Study on Colour Terms](https://www.aclweb.org/anthology/2020.acl-main.584.pdf) ACL2020



#### 🌟🌟🌟 ----F-a-s-h-i-o-n----

(10) Fashion
- [**SIMMC**](https://github.com/facebookresearch/simmc) - Domains include furniture and fashion 🌟🌟🌟, it can be seen as a variant of [multiWOZ](https://github.com/budzianowski/multiwoz) or [schema guided dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue#scheme-representation%5D)
  - [SIMMC 1.0](https://arxiv.org/abs/2006.01460) in Coling2020, [SIMMC 2.0](https://arxiv.org/pdf/2104.08667.pdf), track in [DSTC9](https://dstc9.dstc.community/home) and [DSTC10](https://sites.google.com/dstc.community/dstc10/tracks) 
  - [[Code](https://github.com/facebookresearch/simmc)] 
  - Further papers
    - [A Response Retrieval Approach for Dialogue Using a Multi-Attentive Transformer](https://arxiv.org/abs/2012.08148) second winner DSTC9 SIMMC fashion, [[code](https://github.com/D2KLab/dstc9-SIMMC)]
    - [Overview of the Ninth Dialog System Technology Challenge: DSTC9](https://arxiv.org/pdf/2011.06486.pdf) to better see the winners' models
    - [[Code winner1 TNU](https://github.com/billkunghappy/DSTC_TRACK4_ENTER)](有点乱), [[Code winner2 SU](https://github.com/inkoon/simmc), [[Code other](https://github.com/facebookresearch/simmc/blob/master/DSTC9_SIMMC_RESULTS.md)]

- [**Fashion IQ**](https://sites.google.com/view/cvcreative2020/fashion-iq) in CVPR2020 workshop, [[paper](https://arxiv.org/pdf/1905.12794.pdf)] [[dataset & startkit](https://github.com/XiaoxiaoGuo/fashion-iq)] 

- [**MMD** Towards Building Large Scale Multimodal Domain-Aware Conversation Systems](https://arxiv.org/abs/1704.00200), arXiv 2017, [[code](https://amritasaha1812.github.io/MMD/)], [Multimodal Dialogs (MMD): A large-scale dataset for studying multimodal domain-aware conversations] 2017


### Video
 
(11) 
- [Audio Visual Scene-Aware Dialog Track in DSTC8](http://workshop.colips.org/dstc7/dstc8/Audiovisual_Scene_Aware_Dialog.pdf) [[Paper]((https://ieeexplore.ieee.org/document/8953254)]  [[site]]((https://video-dialog.com/) 
  - [CMU Sinbad’s Submission for the DSTC7 AVSD Challenge]
  - [DSTC8-AVSD: Multimodal Semantic Transformer Network with Retrieval Style Word Generator] 2020
  - [A Simple Baseline for Audio-Visual Scene-Aware Dialog] CVPR2019
- [[TVQA](https://arxiv.org/abs/1809.01696)] [[MovieQA](http://movieqa.cs.toronto.edu/)] [[TGif-QA](https://arxiv.org/abs/1704.04497)]
  - [TVQA+: Spatio-Temporal Grounding for Video Question Answering](https://www.aclweb.org/anthology/2020.acl-main.730.pdf) ACL2020
  - [MultiSubs: A Large-scale Multimodal and Multilingual Dataset] 2021
  - [Adversarial Multimodal Network for Movie Question Answering] 2019
  - [What Makes Training Multi-Modal Classification Networks Hard?] CVPR2020
- [DVD: A Diagnostic Dataset for Multi-step Reasoning in Video Grounded Dialogue] 2021
- video & QA/Dialog papers
  - representation learning
    - VideoBERT: A Joint Model for Video and Language Representation Learning
    - Learning Question-Guided Video Representation for Multi-Turn Video Question Answering ViGil@NeuIPS2019
    - Video Dialog via Progressive Inference and Cross-Transformer EMNLP2019
    - [Multimodal Transformer Networks for End-to-End Video-Grounded Dialogue Systems] ACL2019
    - [Bridging Text and Video: A Universal Multimodal Transformer for Video-Audio Scene-Aware Dialog] 2020
    - [Video-Grounded Dialogues with Pretrained Generation Language Models](https://www.aclweb.org/anthology/2020.acl-main.518.pdf) ACL2020
  - Graph
    - Location-Aware Graph Convolutional Networks for Video Question Answering
    - Object Relational Graph With Teacher-Recommended Learning for Video Captioning
  - Fusion
    - End-to-end Audio Visual Scene-aware Dialog Using Multimodal Attention-based Video Features IEEE2019
    - [See the Sound, Hear the Pixels] IEEE2020
    - [Video Dialog via Multi-Grained Convolutional Self-Attention Context Networks] SIGIR2019
    - [Video Dialog via Multi-Grained Convolutional Self-Attention Context Multi-Modal Networks] IEEE2020
    - [Game-Based Video-Context Dialogue] EMNLP2018
    - [Long-Form Video Question Answering via Dynamic Hierarchical Reinforced Networks] IEEE2019
    - [End-to-End Multimodal Dialog Systems with Hierarchical Multimodal Attention on Video Features] 2018
  

### Charts / figures
(13) [LEAF-QA: Locate, Encode & Attend for Figure Question Answering](https://openaccess.thecvf.com/content_WACV_2020/papers/Chaudhry_LEAF-QA_Locate_Encode__Attend_for_Figure_Question_Answering_WACV_2020_paper.pdf)

### Meme
(14) [MOD Meme incorporated Open Dialogue](https://anonymous.4open.science/r/e7eaef6a-b6d5-47c6-896f-93265a0af4b1/README.md) WeChat conversations with meme / stickers in Chinese language.
- A Multimodal Memes Classification: A Survey and Open Research Issues
- [Learning to Respond with Stickers: A Framework of Unifying Multi-Modality in Multi-Turn Dialog] WWW2020
- [Learning to Respond with Your Favorite Stickers: A Framework of Unifying Multi-Modality and User Preference in Multi-Turn Dialog] 2020
- [The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes] NeuIPS2020

# Survey
[Multimodal Research in Vision and Language: A Review of Current and Emerging Trends](https://arxiv.org/pdf/2010.09522v2.pdf)
[Transformers in Vision: A Survey]
[Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods] 2019

# In general
- Tasks
  - Visual Question Answering, 
  - Visual dialog
  - Visual Commonsense Reasoning, 
  - Image-Text Retrieval, 
  - Referring Expression Comprehension, 
  - Visual Entailment
  - NL+V representation ==> multimodal pretraining
- Issues / topics:
  - text and image bias
  - VL or LV bertologie
  - visual understanding / reasoning / object relation
  - cross-modal text-image relation (attention on interaction)
  - incorporate knowledge / common sense (attention on knowledge)
  - not so much talking about "image retrieval", most of them talk about "image caption", "ground language to image"
- Often used model-elements :
  - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks]() 2015
  - LSTM
  - Transformers
  - Graphs : attention graph, GCN, memory graph .........
- often mentioned approaches:
  - adversial training
  - reinforcement learning
  - graph neural network
  - joint learning / parel / Dual encoder / Dual attention
- my questions
  - what does "adaptive" mean? why everyone likes this specific word?
