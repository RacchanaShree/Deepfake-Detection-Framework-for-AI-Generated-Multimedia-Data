section_15 = """
## 15. BUDGET JUSTIFICATION FOR DIFFERENT HEADS

### 1. Research Staff

**Justification for Zero Budget Allocation:**

This research project does not require dedicated research staff for the following reasons:

- **Principal Investigator Capability:** The Principal Investigator (PI) possesses the necessary technical expertise in deep learning, computer vision, and audio processing to independently conduct all aspects of the research, including model development, training, evaluation, and analysis.

- **Nature of Computational Research:** Unlike field-based or laboratory research requiring multiple personnel for data collection, this is a computational research project where a single researcher can effectively manage all tasks using existing datasets (FakeAVCeleb v1.2).

- **Academic Supervision:** The research will be conducted under faculty supervision at the affiliated institution, providing necessary guidance and oversight without requiring paid research associates or assistants.

- **Cost Optimization:** Eliminating research staff costs allows allocation of resources to critical computational infrastructure (GPU workstations and cloud computing) essential for deep learning model training.

- **Student/Early Career Research:** This project is designed as a student or early-career researcher initiative where the PI will gain hands-on experience across all research phases, from literature review to implementation and publication.

**No Research Associate, Research Assistant, or Field Investigator positions are required.**

---

### 2. Field Work

**Justification for Zero Budget Allocation:**

Field work expenses are not applicable to this research project due to its computational nature:

- **Dataset Availability:** The research utilizes the publicly available FakeAVCeleb v1.2 dataset, which is freely downloadable from online repositories. No primary data collection through field surveys, interviews, or observations is required.

- **Computational Research Methodology:** The entire research workflow—data preprocessing, model development, training, evaluation, and validation—is conducted using computational methods on workstations and cloud infrastructure. No physical field sites or data collection locations are involved.

- **No Geographic Data Collection:** Unlike social science or environmental research requiring field visits, this AI/ML research does not involve traveling to specific locations for data gathering, participant recruitment, or on-site observations.

- **Remote Collaboration:** All research activities, including collaboration with advisors and potential co-authors, can be conducted remotely using digital communication tools.

- **Dataset Characteristics:** The FakeAVCeleb dataset contains pre-recorded video and audio samples that require no additional field-based data augmentation or collection.

**No field work, travel for data collection, or site visits are necessary for this project.**

---

### 3. Equipment and Material

**Total Budget: ₹2,00,000**

This category represents the core infrastructure investment essential for conducting deep learning research. Detailed justification for each component:

#### **A. GPU Workstation (₹80,000)**

**Necessity:**
- Deep learning model training requires GPU acceleration for feasible computation times. CPU-only training would take weeks/months versus hours/days with GPU.
- Essential for model development, prototyping, debugging, and inference testing.
- Enables local experimentation without constant reliance on cloud resources.

**Specification Justification:**
- **RTX 3060 (12GB) or RTX 4060 (8GB):** Mid-range GPUs offering optimal price-performance ratio for research workloads.
- **12GB VRAM:** Sufficient for training hybrid LSTM-Transformer models with batch size 16, as specified in methodology.
- **Local Development:** Reduces cloud costs by enabling local prototyping and smaller-scale experiments.

**Cost Breakdown:**
- GPU Card: ₹45,000-50,000
- Supporting Hardware (CPU, RAM, Motherboard, PSU): ₹30,000-35,000

---

#### **B. Storage: 1TB SSD + 2TB HDD Backup (₹20,000)**

**Necessity:**
- **Dataset Storage:** FakeAVCeleb v1.2 dataset requires approximately 200-300GB storage for video and audio files.
- **Model Checkpoints:** Training generates multiple model checkpoints (5-10GB each) for different epochs and configurations.
- **Preprocessed Data:** Extracted frames, audio segments, and augmented data require additional 100-200GB.
- **Results Archive:** Experimental results, logs, visualizations, and analysis outputs need persistent storage.

**Specification Justification:**
- **1TB NVMe SSD (₹8,000-10,000):** High-speed storage for active datasets and training data, ensuring fast I/O during model training.
- **2TB HDD (₹4,000-5,000):** Cost-effective backup storage for archival of completed experiments, raw datasets, and long-term data retention.

---

#### **C. Cloud GPU Credits (₹1,80,000 over 24 months)**

**Necessity:**
- **Primary Training Infrastructure:** Local GPU workstation insufficient for full-scale model training on complete dataset.
- **Computational Requirements:** Hybrid LSTM-Transformer model training requires 300-400 GPU hours for complete training pipeline including cross-validation and ablation studies.
- **Scalability:** Cloud resources enable parallel experiments and hyperparameter tuning impossible on single local GPU.

**Cost Calculation:**
- **GPU Type:** NVIDIA T4 or V100 instances (cost-effective for research)
- **Hourly Rate:** ₹450-600 per GPU hour (AWS/GCP/Azure student pricing)
- **Total Hours:** 350 hours × ₹500/hour = ₹1,75,000
- **Buffer:** ₹5,000 for additional experiments

**Monthly Breakdown:** ₹1,80,000 ÷ 24 months = ₹7,500/month

**Optimization Strategies:**
- Leveraging student/academic cloud credits (AWS Educate, GCP Education, Azure for Students)
- Using Google Colab Pro+ for supplementary compute
- Mixed-precision training to reduce GPU hours
- Efficient checkpointing to minimize redundant computation

---

#### **D. Cloud Storage & Software Tools (₹20,000 over 24 months)**

**Necessity:**
- **Version Control & Collaboration:** Cloud storage for code versioning, dataset sharing, and collaborative development.
- **Backup & Redundancy:** Off-site backup of critical research data and model weights.
- **Software Subscriptions:** Minimal paid tools where open-source alternatives insufficient.

**Cost Breakdown:**
- **Cloud Storage (Google Drive/AWS S3):** ₹12,000 (500GB-1TB storage)
- **Development Tools:** ₹5,000 (GitHub Pro, specialized tools)
- **Miscellaneous Software:** ₹3,000 (video/audio processing utilities)

**Monthly Breakdown:** ₹20,000 ÷ 24 months = ₹833/month

**Note:** Majority of software stack uses open-source tools (PyTorch, Python, OpenCV, Librosa, Gradio) with zero licensing costs.

---

### **Overall Budget Philosophy:**

The budget is designed with the following principles:
1. **Computational Priority:** Maximum allocation to GPU resources (cloud + local) essential for deep learning research
2. **Cost Optimization:** Leveraging student credits, open-source tools, and efficient training methods
3. **No Redundancy:** Zero allocation to unnecessary staff or field work
4. **Research Focus:** Every rupee allocated directly supports model development, training, or dissemination
5. **Realistic Scope:** Budget aligns with 24-month timeline and achievable research objectives

---

"""

# Read existing file
with open('ICSSR_final_submission.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position to insert (before ## NOTES:)
insert_pos = content.find('## NOTES:')

# Insert Section 15
new_content = content[:insert_pos] + section_15 + content[insert_pos:]

# Write back
with open('ICSSR_final_submission.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Section 15 added successfully!")
