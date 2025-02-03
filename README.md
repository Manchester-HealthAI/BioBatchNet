# BioBatchNet

## Installation

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Manchester-HealthAI/BioBatchNet](https://github.com/Manchester-HealthAI/BioBatchNet
```

### Set Up the Environment

Create a virtual environment and install dependencies using `requirements.yaml`:

#### Using Conda:

```bash
conda env create -f requirements.yaml
conda activate BioBatchNet
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
mv <your-scrna-dataset> Data/gene_data/
```

### Batch effect correction

**For IMC Data**
To process **IMC** data, modify the dataset, network layers, and other parameters in `config_imc.yaml`, then run the following command to train BioBatchNet:
```bash
python IMC.py -c config_imc.yaml
```

✅ **Sample Data Available**  
Sample IMC data is provided in the `Data/IMC` directory. You can directly test the pipeline using:
```bash
python IMC.py -c config_imc.yaml
```

**For scRNA-seq Data**
To process **scRNA-seq** data, modify the dataset, network layers, and other parameters in `config_gene.yaml`, then run the following command to train BioBatchNet:
```bash
python Gene.py -c config_gene.yaml
```

---
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

---
## 📂 Data Download Link

To use BioBatchNet for **batch effect correction**, you need to download the corresponding dataset and place it in the appropriate directory.

### **🔹 Download scRNA-seq Data**
The **scRNA-seq dataset** is available on OneDrive. Click the link below to download:

🔗 [Download scRNA-seq Data](https://livemanchesterac-my.sharepoint.com/:f:/g/personal/haiping_liu_student_manchester_ac_uk/Ep189brW69lJtv4ugZ9QdlkBx23hoFJWbTps_nK9LjZkyw?e=Iz9hOJ)

### **🔹 Download IMC Data**
The **IMC dataset** can be accessed from the **Bodenmiller Group IMC datasets repository**. Visit the link below to explore and download the datasets:

🔗 [IMC Datasets - Bodenmiller Group](https://github.com/BodenmillerGroup/imcdatasets)


## To Do List

- [x] Data download link
- [ ] Checkpoint
- [ ] Benchmark method results

## License

This project is licensed under the MIT License. See the LICENSE file for details.

