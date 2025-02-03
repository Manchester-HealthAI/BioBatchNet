# BioBatchNet

## Installation

### Clone the Repository

First, clone the repository to your local machine:

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

### **For IMC Data**
To process **IMC** data, modify the dataset, network layers, and other parameters in `config_imc.yaml`, then run the following command to train BioBatchNet:
```bash
python IMC.py -c config_imc.yaml
```

✅ **Sample Data Available**  
Sample IMC data is provided in the `Data/IMC` directory. You can directly test the pipeline using:
```bash
python IMC.py -c config_imc.yaml
```

---

### **For scRNA-seq Data**
To process **scRNA-seq** data, modify the dataset, network layers, and other parameters in `config_gene.yaml`, then run the following command to train BioBatchNet:
```bash
python Gene.py -c config_gene.yaml
```


## CPC Usage
CPC uses the embedding of BioBatchNet as the input, and the sample data which is the batch effect corrected embedding of IMMUcan IMC data.
Using the same environments with BioBatchNet, and all the results can be found in the following directory:

```bash
cd CPC/IMC_experiment
```

## To Do List

- [ ] Data download link
- [ ] Checkpoint
- [ ] Benchmark method results

## License

This project is licensed under the MIT License. See the LICENSE file for details.

