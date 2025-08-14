---

## Datasets
The pipeline supports:
- **CERT Insider Threat Dataset** (logon, file, device logs)
- **LANL Authentication Dataset** (Kerberos authentication events)
- **Enron Email Dataset** (emails with metadata and body content)

**Sources:**
- CERT: https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
- LANL: https://csr.lanl.gov/data/cyber1/
- Enron: https://www.cs.cmu.edu/~enron/

---

## Setup & Execution (Google Colab)

### 1. Open in jypter or any python IDE
- Create a **new notebook**.

### 2. Install Dependencies
```python
!pip -q install pandas numpy scikit-learn matplotlib torch --extra-index-url https://download.pytorch.org/whl/cpu
