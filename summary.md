Multi-modal dialogue systems consist of those dialogue systems that deal with multi-modal inputs and outputs, besides of textual modality,
audio, visual or audio-visual features are taken consideration as multi-modalities. In our very simple and brief survey, we would like to concentrate
on visual dialogue systems at the first time. 我这里参考了比较多[这篇survey](https://arxiv.org/pdf/2010.09522.pdf).

Visual dialogue system can be seen as a variant subtask of Visual Question Answering (VQA) which remains to be the most popular task as well as Visual Captioning
across various visual-linguistic tasks in the research community. It comprises of VQA in dialogue (VQADi -- Visual Dialogue v1), and VQG in dialogue (VQGDi -- GuessWhich Task)
wherein the main goal is to automate machine conversations about images with humans. 

The referred survey defines the formats of output in purpose of distinguish "generation task" and "classification task": 

> The output of the learnt mapping *f* could either belong to a set
of possible answers in which case we refer this task format
as MCQ, or could be arbitrary in nature depending on the
question in which we can refer to as free-form. We regard the
more generalized free-form VQA as a generation task, while
MCQ VQA as a classification task where the model predicts
the most suitable answer from a pool of choices. 

Generally, classical VQA datasets/tasks like VQA 1.0 & 2.0 apply MCQ format of output while visual dialogue tasks and VQG tasks on VQA 2.0 have *free form* format of output. Specifically, the survey takes out separately the Visual Commensense Reasoning as a popular task in parallel to VQA tasks, which aims to develop higherorder cognition in vision systems and commonsense reasoning of the world so that they can provide justifications to their answers.

I have to mention VQA because it's a group of classical and living datasets and tasks that are incontourable before talking about Visual Dialogue, and many of the most popular issues and topics and methods affect also those in visual dialogues, like reasoning, deep understanding, modality bias, etc., while in special, the works about VCR show a special kind of enthusiasm in using bertology methods like Vi-Bert, VisualBERT, VL-Bert,
KVL-BERT, etc. (A paper about [The Exploration of the Reasoning Capability of BERT in Relation Extraction](https://ieeexplore.ieee.org/document/9202183) but it was published only in 2020. So why suddenly a bunch of Bertologies in Visual-Language research keeps to be a mistery for me...) Of course, in visual dialogue tasks we have bertology methods, like VD-BERT. 

| Article        | Dataset           | Visual Encoder  | Language Model | Encoder | Decoder |
| ------------- |:-------------:|:---------------:|:------------:|:----------:|:---------:|
| [Visual dialog](https://arxiv.org/pdf/1611.08669.pdf)| VisDial v1.0 | VGG16 | 2 diff LSTMs; dialog-RNN + Attention + LSTM ; LSTM | Late fusion ; HRE ; MN| LSTM, Softmax |
| [IQA Synergistic](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Image-Question-Answer_Synergistic_Network_for_Visual_Dialog_CVPR_2019_paper.pdf)| VisDial v1.0 |Faster-RCNN, CNN  | 2 diff LSTMs | MFB ; (discriminative model: in primary stage, answers are also encoded by LSTM)| softmax; (generative model: answer decoded by LSTM)|
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
| [LTMI](https://arxiv.org/pdf/1911.11390.pdf) | VisDial v1.0 |  |  | 
| [LF]      | centered      |    |  |  | fusion by concat |  |
