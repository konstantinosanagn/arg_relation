# Project Documentation

This repository contains files and scripts to construct and run experiments on Llama and other models. Below is a guide to help you navigate and utilize the project effectively.

---

## Prerequisites

### 1. **Miniconda**
Ensure that [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed on your system. It is used to manage the project's environment and dependencies.

### 2. **Requirements**
Install the required Python libraries by running:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

The \`requirements.txt\` file contains all necessary dependencies for running the models.

---

## Directory Structure and Key Files

### Projects Directory
Most files in this directory are components for constructing and running experiments on **Llama**. For a clear example of how the models are integrated, refer to:

- **\`link_pred_mistral.py\`**  
  This script demonstrates running **Mistral** on the link prediction task. It uses the Hugging Face \`transformers\` library to pipe the model.

- **/`vicuna/`**
  This directory contains scripts that run additional models from scratch. Very useful!

**Note:**  
For **Llama**, the setup is slightly different. The models were obtained through Meta and downloaded onto the server. While the approach differs, the fundamental process of loading a model is similar to Mistral.

### Downloading Llama Models
If you're eager to download the Llama models, consult the **additional \`README.txt\`** located in the directory. It provides detailed instructions for obtaining and setting up the models.

---

## Dataset Information

The datasets are structured as follows:

1. **Original Dataset:**  
   Located in \`/parklab/data\`. This is the unprocessed dataset.
   
2. **Parsed Dataset:**  
   Found in \`/argument_relation/data\`. This directory contains:
   - \`train\` set
   - \`validation\` set
   - \`test\` set  

**Note:**  
The train, validation, and test sets were randomly generated from the original dataset. To ensure consistency, avoid modifying the \`test\` set during model training or testing.

---

## Logging and Debugging

Unfortunately, there is minimal logging available. However, print statements have been added throughout the codebase to help guide you during execution and debugging.

---

## Final Notes

This repository represents a robust starting point for experimenting with advanced models. Be sure to familiarize yourself with the file structure and the available documentation.

Good luck with your project! 🚀
