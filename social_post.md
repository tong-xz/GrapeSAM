# GrapeSAM — Social Media Posts

**Links**
- 📄 Paper: https://arxiv.org/abs/2605.24353
- 💻 Code: https://github.com/tong-xz/GrapeSAM
- 📦 Dataset: https://huggingface.co/datasets/XZhi/ViViD-5k

**Images to attach**
- Primary: `assets/teaser.png` — pipeline narrative (cluster mask → per-berry detection → localization)
- Secondary: `assets/teaser2.png` — gallery grid showing generalization across clusters

---

## 🐦 X / Twitter

### Single post

🍇 We present **GrapeSAM** — a unified computer-vision pipeline for grape cluster analysis. From one cluster image: whole-cluster segmentation, per-berry localization, and cluster-closure computation in a single framework. Built on Mask2Former + SAM.

We also release **ViViD-5k**: 5,000 vineyard images, 13 grape varieties, 648K+ berry keypoints, 18K+ cluster masks. 👇

📄 https://arxiv.org/abs/2605.24353
💻 https://github.com/tong-xz/GrapeSAM
📦 https://huggingface.co/datasets/XZhi/ViViD-5k

#ComputerVision #DeepLearning #AgTech

### Thread version

1/ 🍇 We present **GrapeSAM**, a unified pipeline for grape cluster analysis from a single image: whole-cluster segmentation → per-berry localization → cluster-closure computation. (1/4)

2/ Method: Mask2Former segments the cluster; Segment Anything together with a point-localization model resolves individual berries. The pipeline is modular — each stage runs independently. (2/4)
[attach teaser.png]

3/ We also release **ViViD-5k**, a large-scale vineyard dataset for grape cluster analysis:
• 5,000 images across 13 grape varieties
• 648,000+ annotated berry-centroid keypoints
• 18,000+ cluster instance masks & bboxes (3/4)
[attach teaser2.png]

4/ Code, dataset, and paper are open:
📄 https://arxiv.org/abs/2605.24353
💻 https://github.com/tong-xz/GrapeSAM
📦 https://huggingface.co/datasets/XZhi/ViViD-5k
Work done @Cornell — huge thanks to my advisor @[handle] & collaborators. 🙏 (4/4)

---

## 💼 LinkedIn

🍇 **Introducing GrapeSAM: a unified computer-vision pipeline for grape cluster analysis.**

We present GrapeSAM, a framework that, from a single image of a grape cluster, performs three tasks in one pipeline:
• Whole-cluster segmentation (Mask2Former)
• Per-berry detection and localization (Segment Anything + a point-localization model)
• Berry counting and cluster-closure computation

Berry counting and cluster assessment are labor-intensive and difficult to scale in viticulture. GrapeSAM addresses this with an end-to-end, modular pipeline that generalizes across diverse cluster morphologies, colors, and densities.

Alongside the method, we release **ViViD-5k**, a large-scale vineyard image dataset for grape cluster analysis:
• 5,000 images spanning 13 grape varieties
• 648,000+ annotated berry-centroid keypoints
• 18,000+ grape cluster instance masks & bounding boxes

📄 Paper: https://arxiv.org/abs/2605.24353
💻 Code: https://github.com/tong-xz/GrapeSAM
📦 Dataset: https://huggingface.co/datasets/XZhi/ViViD-5k

This work was carried out during my time at Cornell University. I'm sincerely grateful to my advisor @[Advisor Name] and my collaborators for their guidance and support throughout this project. 🙏 (Built on Segment Anything and Mask2Former.)

#ComputerVision #DeepLearning #AgTech #Agriculture #MachineLearning #OpenScience

---

## How to tag / @-mention collaborators

### X / Twitter
- Type `@` followed by their **handle** (e.g. `@jane_doe`), not their display name. An autocomplete dropdown appears — select the right account so it links.
- Mentions in the **main tweet body** count toward the 280-character limit. To avoid clutter, add a closing line like: `Work with @collab1 @collab2 @collab3`.
- Tip: putting mentions as the **first reply** in the thread (instead of the main post) keeps the headline clean and still notifies everyone.
- You can also tag up to **10 accounts directly on the image** (the photo tag does not use characters) — click the image after attaching → "Tag people".
- Tag the institutions/labs too if they have accounts (e.g. `@MetaAI`).

### LinkedIn
- Type `@` then start typing the person's **name**; wait for the dropdown and click their profile so it turns into a blue link. If you don't select from the dropdown, it posts as plain text and does NOT notify them.
- You can mention both **people** and **company/organization Pages** (e.g. your university or lab) the same way.
- Only people who are connections (or whose name resolves in the dropdown) reliably get notified — double-check each one turned blue before posting.
- Best practice: weave mentions into a sentence near the end, e.g. *"Joint work with [Name], [Name], and [Name] at [University Page]."*
- After posting, you can still edit the post to fix or add a mention if one didn't link.

### General etiquette
- Confirm collaborators are OK being tagged before posting (some prefer not to be).
- Get exact handles/profile URLs in advance so the dropdown matches on the first try.
- Tag the **paper's corresponding author / first author**, **co-authors**, your **lab/university Page**, and relevant orgs (dataset host, funding bodies) where appropriate.
