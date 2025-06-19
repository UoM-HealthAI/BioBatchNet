# BioBatchNet

## Installation
### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Manchester-HealthAI/BioBatchNet](https://github.com/Manchester-HealthAI/BioBatchNet
```

### Set Up the Environment

Create a virtual environment and install dependencies using `environment.yml`:

#### Using Conda:

```bash
conda env create -f environment.yml
conda activate bbn
```

## BioBatchNet Usage

### Enter BioBatchNet
```bash
cd BioBatchNet
```

### Construct dataset
For the IMC dataset, place the dataset inside:

```bash
mv <your-imc-dataset> Data/IMC/
```

For scRNA-seq data, create a folder named `gene_data` inside the `Data` directory and place the dataset inside:

```bash
mkdir -p Data/gene_data/
mv <your-scrna-dataset> Data/scRNA-seq/
```

### Batch effect correction

**For IMC Data**
To process **IMC** data, run the following command to train BioBatchNet:
```bash
python imc.py -c config/IMC/IMMUcan.yaml
```

**For scRNA-seq Data**
To process **scRNA-seq** data, modify the dataset, run the following command to train BioBatchNet:
```bash
python scrna.py -c config/IMC/macaque.yaml
```

## CPC Usage

CPC utilizes the **embedding output from BioBatchNet** as input. The provided sample data consists of the **batch effect corrected embedding of IMMUcan IMC data**.

To use CPC, ensure you are running in the **same environment** as BioBatchNet.  
All experiment results can be found in the following directory:

```bash
cd CPC/IMC_experiment
```

✅ **Key Notes**:  
- CPC requires embeddings from BioBatchNet as input.  
- Sample data includes batch-corrected IMMUcan IMC embeddings.  
- Ensure the **same computational environment** as BioBatchNet before running CPC.  

## 📂 Data Download Link

To use BioBatchNet for **batch effect correction**, you need to download the corresponding dataset and place it in the appropriate directory.

### **🔹 Download scRNA-seq Data**
The **scRNA-seq dataset** is available on OneDrive. Click the link below to download:

🔗 [Download scRNA-seq Data](https://drive.google.com/drive/folders/1m4AkNc_KMadp7J_lL4jOQj9DdyKutEZ5?usp=sharing)

### **🔹 Download IMC Data**
The **IMC dataset** can be accessed from the **Bodenmiller Group IMC datasets repository**. Visit the link below to explore and download the datasets:

🔗 [IMC Datasets - Bodenmiller Group](https://github.com/BodenmillerGroup/imcdatasets)


## To Do List

- [x] Data download link
- [ ] Checkpoint
- [ ] Benchmark method results

## License

This project is licensed under the MIT License. See the LICENSE file for details.

