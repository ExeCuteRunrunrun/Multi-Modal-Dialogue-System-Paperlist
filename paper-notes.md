# Multi-Modal-Dialgoue-System-Paperlist

This is a paper list for the multimodal dialogue systems topic.

**Keyword**: Multi-modal, Dialogue system, visual, conversation

# Paperlist

## Dataset & Challenges

### Images

[GuessWhat?!](https://openaccess.thecvf.com/content_cvpr_2017/html/de_Vries_GuessWhat_Visual_Object_CVPR_2017_paper.html) Visual Object Discovery Through Multi-Modal Dialogue in CVPR2017, a two-player guessing game (1 oracle & 1 questioner). [Code](https://github.com/GuessWhatGame/guesswhat)
- **Data and task**: 
  - **Data**: images are from **MS COCO dataset**, questioners & oracles are from **Amazon Mechanical Turk**. Players are asked to shorten their dialogues to speed up the game (and therefore maximize their gains). The goal of the game is to **locate an unknown object** in a **rich image scene** (meaning that there're several objects in an image / photo) by asking a sequence of questions, eg. After a sequence of n questions (of yes / no / NA), it becomes possible to locate the object (highlighted by a green mask). Once the questioner has gathered enough evidence to locate the object, they notify the oracle that they are ready to guess the object. We then reveal the list of objects, and if the questioner picks the right object, we consider the game successful (*recall@k ??*).
  - **Task**: The **oracle task** requires to produce a yes-no answer for any object within a picture given a natural language question. The  **questioner task** is divided into two different sub-tasks that are trained independently: The **Guesser** must predict the correct object
O_correct from the set of all objects O given an image I and a sequence of questions and answers D_J . The **Question Generator** must produce a new question q_T+1 Given an image I and a sequence of T questions and answers D_≤T .
- **Problematic**:  How to create models that understand natural language descriptions and ground them in the visual world. Higher-level image understanding, like spatial reasoning and language grounding, is required to solve the proposed task.
- **Baseline model**: 
  - **Oracle baseline**: a classification problem (yes/no/NA). Embedding = Image (VGG16) + Question (LSTM) + Crop (VGG16) + Spatial information (bbox) + Object Category taxonomy; MLP ; cross-entropy loss. **Reflection**  In general, we expect the object crop to contain additional information, such as color information, beside the object class. However, we find that the object category outperforms the object crop embedding. This might be partly due to the imperfect feature extraction from the crops.
  - **Guesser**: a classification problem (among a list of objects). Embedding1 = Question(LSTM/HRED) + Image(VGG16), Embedding2 = Objects (MLP(Spatial+Category)), then dot product; **Reflection** In general, we find that including VGG features does not improve the performance of the LSTM and HRED models. We hypothesize that the VGG features are a too coarse representation of the image scene, and that most of the visual information is already encoded in the question and the object features (???)
  - **Question Generator**: encoder-decoder structure: P(q_2|q_1,a_1,Image(vgg)), We use our questioner model to first generate a question which is then answered by the oracle model. We repeat this procedure 5 times to obtain a dialogue. We then use the best performing guesser model to predict the object and report its error as the metric for the QGEN model. **Reflection** using the Oracle’s answers while training the Question Generator introduces additional errors than using Ground Truth answers.
- **Further papers & models:
  - **Paper**:[End-to-end optimization of goal-driven and visually grounded dialogue systems](https://arxiv.org/abs/1703.05423) Reinforcement Learning applied to GuessWhat?! based on **policy gradient algorithms**. **Problematic**: Encoder-Decoder structure's drawbacks: 1. vast action space vs unseen scenarios, inconsistent dialogues 2. supervised learning in dialog systems does not account for the intrinsic planning problem, especially in task-oriented dialogues. 3. difficult in naturally integrating external contexts (common ground) 4. difficult in dialogue evaluation. **?quote?** In addition, successful applications of the RL framework to dialogue often rely on a predefined structure of the task, such as slot-filling tasks [Williams and Young, 2007] where the task can be casted as filling in a form. **Proposed model** kan bu dong


[ReferIt](http://tamaraberg.com/referitgame/) [paper](http://tamaraberg.com/papers/referit.pdf) in EMNLP2014
- **Task**: 2 players game, 1 give referring expressions of an object in the image and the other should label it on the image. 
- **Data processing**: To avoid hand annotations, use attributes for referring expressions and a template-based parser. 
- **Model** to generate referring expressions (a set of attributes, not in natural language) too much mathematics I'm blind.

[Image Captioning] generating natural language descriptions of images.
- **Data and task**: [MS Coco](https://arxiv.org/pdf/1405.0312.pdf) Images + captions (but captions are single words not sentences)
- **Further papers & models**:
  - [Object relation transformer](https://papers.nips.cc/paper/2019/file/680390c55bbd9ce416d1d69a9ab4760d-Paper.pdf) in NIPS2019 **Model** always Encoder-Decoder structure but this time a Transformer (!), given an input image, use object detection model to get appearance features and geometry features (actually meaning a bounding box and its object and features), take these as inputs of the Transformer 

[Visual QA](https://visualqa.org/workshop.html) VQA datasets in CVPR2021,2020,2019,etc
- **Data and task**: 
  - [VQA](https://visualqa.org/challenge) datasets [1.0](http://arxiv.org/abs/1505.00468) [2.0](https://arxiv.org/abs/1612.00837)
  - [TextVQA](https://textvqa.org/paper) TextVQA requires models to read and reason about text in an image to answer questions based on them. In order to perform well on this task, models need to first detect and read text in the images. Models then need to reason about this to answer the question. 
  - [TextCap](https://arxiv.org/abs/2003.12462) TextCaps requires models to read and reason about text in images to generate captions about them. Specifically, models need to incorporate a new modality of text present in the images and reason over it and visual content in the image to generate image descriptions.
- **Baseline models**
- **Proposed papers & models**:



[Visual Dialog](https://visualdialog.org/#:~:text=Visual%20Dialog%20is%20a%20novel,has%20to%20answer%20the%20question.) Open-domain dialogs & given an image, a dialog history, and a follow-up question about the image, the task is to answer the question. [[VisDial v1.0 dataset](https://visualdialog.org/data)] [[Paper](https://arxiv.org/abs/1611.08669)] [[Code diverse questions](https://github.com/vmurahari3/visdial-diversity)] [[Code rl](https://github.com/batra-mlp-lab/visdial-rl)] [[Code collect chat](https://github.com/batra-mlp-lab/visdial-amt-chat)] 🌟🌟

[SIMMC](https://github.com/facebookresearch/simmc) Situated and Interactive Multimodal Conversations track in [DSTC9](https://dstc9.dstc.community/home) and [DSTC10](https://sites.google.com/dstc.community/dstc10/tracks) by Facebook [[Paper](https://arxiv.org/abs/2006.01460)] Domains include furniture and fashion 🌟🌟🌟

[Fashion IQ](https://sites.google.com/view/cvcreative2020/fashion-iq) in CVPR2020 workshop, [[paper](https://arxiv.org/pdf/1905.12794.pdf)] [[dataset & startkit](https://github.com/XiaoxiaoGuo/fashion-iq)]

### Video

[AVSD - Audio Visual Scene-Aware Dataset](https://video-dialog.com/) was used in [DSTC7](http://workshop.colips.org/dstc7/) and [DSTC8](https://sites.google.com/dstc.community/dstc8/tracks) The task is to build a system that generates response sentences in a dialog about an input VIDEO. The data collection paradigm is similar to VisDial

Related : [[TVQA](https://arxiv.org/abs/1809.01696)] [[MovieQA](http://movieqa.cs.toronto.edu/)] [[TGif-QA](https://arxiv.org/abs/1704.04497)]

### Meme

[MOD Meme incorporated Open Dialogue](https://anonymous.4open.science/r/e7eaef6a-b6d5-47c6-896f-93265a0af4b1/README.md) WeChat conversations with meme / stickers in Chinese language.







