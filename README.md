# GRUWITHAUGMENTATION

## Proposed Method
<div style="font-size: 2em;">
We apply a modified version of GRU with various Data Augmentation Methods.
 </div> 
 
---

<div style="font-size: 2em;">
  Firstly, our modification entails the integration of information from both the current ( t ) and preceding ( t-1 ) time steps. 
  
  By updating the GRU Cell design to include information from two sequential time points, we aim to improve the learning of time sequences.  
  
</div>
  <img src="https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/7e406e26-b0d5-4f66-9d71-0ab9e0b88eb5" alt  = "momdelflow">
  </div>

---

<div style="font-size: 2em;">
  And we propose to use various data augmentation techniques to effectively train the feature extractor to learn more generalized and discriminative features and to enhance the model's robustness.

  To start with, we **add Gaussian noise** to the feature data.
  
</div>

<div style="text-align:center;">
  <img src="https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/75d2ebf3-aa45-4332-93af-5ebce2a7df93" alt="gaussian">
</div>

---

<div style="font-size: 2em;">
  and then apply the shifting technique that shifts the first 10 steps and the last 10 steps of this data forward or backward.
  
</div>

<div style="text-align:center;">
  <img src="https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/c7788a67-71aa-4460-8b46-e24703fb6cc9" alt="shifting">
</div>

---

<div style="font-size: 2em;">
  Lastly, we added Cutmix which is traditionally known as an image augmentation technique.
  
</div>

<div style="text-align:center;">
  <img src="https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/ecc190fe-62b7-4ad8-aea3-3ea8c9297c11" alt="cutmix">
</div>

---

<div style="font-size: 2em;">
  In conclusion, the overall Data Augmentation method is shown below.
  
</div>

<div style="text-align:center;">
  <img src="https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/652ba656-a7f2-4fea-86fd-583c39e9bae7" alt="augflow">
</div>

---

<div style="font-size: 2em;">
  Our proposed method shows great improvement compared with State-of-the-Art (SOTA) with 96.76% accuracy.  
</div>

<div style="text-align:center;">
  <img src="https://github.com/harikang/GRUWITHAUGMENTATION/assets/138555992/6af44e2d-f7b2-4817-bd31-6ef9ab4b2bbb" alt="confusionmatrix">
</div>
