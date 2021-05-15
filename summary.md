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


